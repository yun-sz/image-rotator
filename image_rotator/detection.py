from typing import Optional, List, Dict, Tuple

import cv2
import numpy as np
from PIL import Image
import pytesseract

from .config import (
    OSD_CONFIG, PIPELINE_WEIGHTS,
    OSD_EARLY_STOP_MIN_RESULTS, OSD_EARLY_STOP_CONF,
    OSD_TRUST_MIN_RESULTS, OSD_TRUST_AGREEMENT, OSD_TRUST_OSD_CONF,
    OSD_LOW_CONFIDENCE, MIN_CONFIDENCE_RATIO
)
from .types import OSDResult, OrientationResult
from .preprocessing import resize_for_osd, resize_for_ocr, OSD_PIPELINES
from .ocr import compute_text_quality_multi


def run_osd(image: np.ndarray, preprocessing_name: str) -> Optional[OSDResult]:
    try:
        if len(image.shape) == 2:
            pil_image = Image.fromarray(image)
        else:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        osd = pytesseract.image_to_osd(
            pil_image,
            output_type=pytesseract.Output.DICT,
            config=OSD_CONFIG
        )

        rotation = osd.get('rotate', 0) % 360
        confidence = float(osd.get('orientation_conf', 0))

        if rotation not in [0, 90, 180, 270]:
            rotation = round(rotation / 90) * 90 % 360

        return OSDResult(
            rotation=rotation,
            confidence=confidence,
            preprocessing=preprocessing_name
        )
    except Exception:
        return None


def detect_with_pipelines(image: np.ndarray) -> OrientationResult:
    results: List[OSDResult] = []
    image_resized = resize_for_osd(image)

    for name, preprocess_func in OSD_PIPELINES:
        try:
            processed = preprocess_func(image_resized)
            result = run_osd(processed, name)
            if result is not None:
                results.append(result)
                if len(results) >= OSD_EARLY_STOP_MIN_RESULTS:
                    first_rot = results[0].rotation
                    if all(r.rotation == first_rot for r in results):
                        avg_conf = sum(r.confidence for r in results) / len(results)
                        if avg_conf >= OSD_EARLY_STOP_CONF:
                            break
        except Exception:
            continue

    if not results:
        return OrientationResult(
            rotation=0,
            confidence=0,
            method="NoDetection",
            votes={0: 0, 90: 0, 180: 0, 270: 0},
            details=[]
        )

    votes: Dict[int, float] = {0: 0, 90: 0, 180: 0, 270: 0}
    vote_counts: Dict[int, float] = {0: 0, 90: 0, 180: 0, 270: 0}

    for r in results:
        weight = PIPELINE_WEIGHTS.get(r.preprocessing, 1.0)
        votes[r.rotation] += r.confidence * weight
        vote_counts[r.rotation] += weight

    best_rotation = max(votes, key=votes.get)
    total_votes = len(results)
    winning_votes = vote_counts[best_rotation]

    agreement_ratio = winning_votes / total_votes
    winning_results = [r for r in results if r.rotation == best_rotation]
    avg_confidence = sum(r.confidence for r in winning_results) / len(winning_results)

    other_rotations = [r for r in votes if r != best_rotation]
    second_rotation = max(other_rotations, key=lambda x: votes[x])
    margin = votes[best_rotation] - votes[second_rotation]

    final_confidence = (
        avg_confidence * 0.4 +
        agreement_ratio * 100 * 0.4 +
        min(margin / (total_votes + 1), 30) * 0.2
    )

    return OrientationResult(
        rotation=best_rotation,
        confidence=final_confidence,
        method=f"OSD({winning_votes}/{total_votes})",
        votes={k: int(v) for k, v in vote_counts.items()},
        details=results
    )


def _rotate_cv2(image: np.ndarray, rotation: int) -> np.ndarray:
    if rotation == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if rotation == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if rotation == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


def _compare_orientations(image: np.ndarray, rotations: List[int]) -> Tuple[int, Dict[int, float]]:
    base = resize_for_ocr(image)
    scores = {rot: compute_text_quality_multi(_rotate_cv2(base, rot)) for rot in rotations}
    return max(scores, key=scores.get), scores


def detect_orientation(image: np.ndarray) -> OrientationResult:
    result = detect_with_pipelines(image)

    total_votes = len(result.details)
    if total_votes == 0:
        best, _ = _compare_orientations(image, [0, 90, 180, 270])
        result.rotation = best
        result.method = "FullOCR"
        return result

    max_votes = max(result.votes.values())
    vote_agreement = max_votes / total_votes
    winning_confs = [r.confidence for r in result.details if r.rotation == result.rotation]
    avg_osd_conf = sum(winning_confs) / len(winning_confs) if winning_confs else 0.0

    vote_conf: Dict[int, float] = {0: 0.0, 90: 0.0, 180: 0.0, 270: 0.0}
    for r in result.details:
        vote_conf[r.rotation] += r.confidence

    # Strong OSD agreement
    if (
        total_votes >= OSD_TRUST_MIN_RESULTS and
        total_votes == len(OSD_PIPELINES) and
        vote_agreement >= OSD_TRUST_AGREEMENT and
        avg_osd_conf >= OSD_TRUST_OSD_CONF
    ):
        result.method += "+OSD-strong"
        return result

    # Split votes
    if vote_agreement < 0.5:
        rotations = sorted(
            [0, 90, 180, 270],
            key=lambda rot: (result.votes.get(rot, 0), vote_conf[rot]),
            reverse=True
        )

        top_votes = [result.votes.get(rotations[i], 0) for i in range(min(3, len(rotations)))]

        if total_votes >= 3 and len(set(top_votes)) == 1:
            best, scores = _compare_orientations(image, [0, 90, 180, 270])
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            if sorted_scores[0][1] > 0:
                ratio = (sorted_scores[0][1] - sorted_scores[1][1]) / sorted_scores[0][1]
                if ratio >= MIN_CONFIDENCE_RATIO:
                    result.rotation = best
                    result.method += "+FullOCR"
            return result

        best, scores = _compare_orientations(image, rotations[:2])
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        if sorted_scores[0][1] > 0:
            ratio = (sorted_scores[0][1] - sorted_scores[1][1]) / sorted_scores[0][1]
            if ratio >= MIN_CONFIDENCE_RATIO:
                result.rotation = best
                result.method += "+OCR-Top2"
        return result

    # Low confidence mixed axis
    if avg_osd_conf < OSD_LOW_CONFIDENCE:
        h_votes = result.votes.get(0, 0) + result.votes.get(180, 0)
        v_votes = result.votes.get(90, 0) + result.votes.get(270, 0)
        if h_votes > 0 and v_votes > 0:
            best_h = 0 if result.votes.get(0, 0) >= result.votes.get(180, 0) else 180
            best_v = 90 if result.votes.get(90, 0) >= result.votes.get(270, 0) else 270
            best, scores = _compare_orientations(image, [best_h, best_v])
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            if sorted_scores[0][1] > 0:
                ratio = (sorted_scores[0][1] - sorted_scores[1][1]) / sorted_scores[0][1]
                if ratio >= MIN_CONFIDENCE_RATIO:
                    result.rotation = best
                    result.method += "+OCR-Axis"
                    return result

    # Verify 0/180 or 90/270
    if result.rotation in [0, 180]:
        pair = [0, 180]
    else:
        pair = [90, 270]

    best, scores = _compare_orientations(image, pair)
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    if sorted_scores[0][1] > 0:
        ratio = (sorted_scores[0][1] - sorted_scores[1][1]) / sorted_scores[0][1]
        if ratio >= MIN_CONFIDENCE_RATIO:
            result.rotation = best
            result.method += f"+OCR({pair[0]}/{pair[1]},r={ratio:.2f})"
            return result

    # Trust OSD
    if result.rotation in [0, 180]:
        result.rotation = 0 if result.votes.get(0, 0) >= result.votes.get(180, 0) else 180
    else:
        result.rotation = 90 if result.votes.get(90, 0) >= result.votes.get(270, 0) else 270
    result.method += "+OSD-trusted"

    return result
