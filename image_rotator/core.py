import time
from typing import Tuple

import numpy as np

from .types import ProcessingResult
from .preprocessing import is_blank_page, rotate_image
from .detection import detect_orientation
from .ocr import extract_text


def process_image(image: np.ndarray) -> Tuple[np.ndarray, ProcessingResult]:
    start = time.perf_counter()

    try:
        if is_blank_page(image):
            return image.copy(), ProcessingResult(
                success=True,
                rotation=0,
                confidence=100,
                method="BlankPage",
                time_ms=(time.perf_counter() - start) * 1000,
                ocr_text=""
            )

        result = detect_orientation(image)
        rotated = rotate_image(image, result.rotation)
        ocr_text = extract_text(rotated)

        return rotated, ProcessingResult(
            success=True,
            rotation=result.rotation,
            confidence=result.confidence,
            method=result.method,
            time_ms=(time.perf_counter() - start) * 1000,
            ocr_text=ocr_text
        )

    except Exception as e:
        return image.copy(), ProcessingResult(
            success=False,
            rotation=0,
            confidence=0,
            method="Error",
            time_ms=(time.perf_counter() - start) * 1000,
            error=str(e)
        )
