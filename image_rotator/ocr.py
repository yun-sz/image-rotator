from typing import List

import cv2
import numpy as np
from PIL import Image
import pytesseract

from .config import (
    OCR_CONFIG, COMMON_WORDS, SYMMETRIC_WORDS,
    NUMBER_PATTERN, DATE_PATTERN
)
from .preprocessing import to_grayscale, preprocess_clahe, OCR_PIPELINES


def extract_text(image: np.ndarray) -> str:
    try:
        gray = to_grayscale(image)
        enhanced = preprocess_clahe(gray)
        pil_img = Image.fromarray(enhanced)
        return pytesseract.image_to_string(pil_img, config=OCR_CONFIG).strip()
    except Exception:
        return ""


def compute_text_quality(image: np.ndarray) -> float:
    try:
        gray = to_grayscale(image)
        pil_img = Image.fromarray(gray)

        word_data = pytesseract.image_to_data(
            pil_img,
            config=OCR_CONFIG,
            output_type=pytesseract.Output.DICT
        )

        score = 0.0
        high_conf_words = 0
        total_words = 0
        texts: List[str] = []

        data_texts = word_data.get('text', [])
        data_confs = word_data.get('conf', [])

        for i, raw_text in enumerate(data_texts):
            text = raw_text.strip()
            if not text:
                continue

            texts.append(text)
            total_words += 1

            try:
                conf_val = float(data_confs[i])
            except Exception:
                conf_val = -1.0

            if conf_val > 70:
                high_conf_words += 1
                score += conf_val * 0.5
                if text.upper() in COMMON_WORDS:
                    score += 100
            elif conf_val > 40:
                score += conf_val * 0.2
                if text.upper() in COMMON_WORDS:
                    score += 30

        if not texts:
            return 0.0

        full_text = " ".join(texts)

        token_set = {t.upper() for t in texts}
        matched_words = (token_set - SYMMETRIC_WORDS) & COMMON_WORDS
        score += len(matched_words) * 80

        score += len(NUMBER_PATTERN.findall(full_text)) * 15
        score += len(DATE_PATTERN.findall(full_text)) * 30

        if total_words > 0:
            score += (high_conf_words / total_words) * 200

        score += sum(1 for c in full_text if c.isalnum()) * 0.5

        allowed = ".,;:-/()[]{}$%@#&*+='\""
        score -= sum(1 for c in full_text if not c.isalnum() and not c.isspace() and c not in allowed) * 5
        score -= sum(1 for w in texts if len(w) == 1 and not w.isdigit()) * 10

        return max(0.0, score)
    except Exception:
        return 0.0


def compute_text_quality_multi(image: np.ndarray) -> float:
    scores = []

    for _, preprocess in OCR_PIPELINES:
        try:
            processed = preprocess(image)
            scores.append(compute_text_quality(processed))
        except Exception:
            continue

    return max(scores) if scores else compute_text_quality(image)
