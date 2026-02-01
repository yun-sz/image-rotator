import cv2
import numpy as np

from .config import OSD_MAX_DIM, OSD_MIN_DIM, OSD_MAX_UPSCALE, OCR_MAX_DIM, BLANK_PAGE_THRESHOLD


def is_blank_page(image: np.ndarray, threshold: float = BLANK_PAGE_THRESHOLD) -> bool:
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    content_ratio = np.sum(binary > 0) / binary.size

    return content_ratio < threshold


def to_grayscale(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.copy()


def preprocess_otsu(image: np.ndarray) -> np.ndarray:
    gray = to_grayscale(image)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def preprocess_adaptive_gaussian(image: np.ndarray) -> np.ndarray:
    gray = to_grayscale(image)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 10)


def preprocess_clahe(image: np.ndarray) -> np.ndarray:
    gray = to_grayscale(image)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def preprocess_clahe_otsu(image: np.ndarray) -> np.ndarray:
    enhanced = preprocess_clahe(image)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def preprocess_denoise_otsu(image: np.ndarray) -> np.ndarray:
    gray = to_grayscale(image)
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def preprocess_sharpen_otsu(image: np.ndarray) -> np.ndarray:
    gray = to_grayscale(image)
    blurred = cv2.GaussianBlur(gray, (0, 0), 3)
    sharpened = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
    _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def preprocess_morphological(image: np.ndarray) -> np.ndarray:
    binary = preprocess_otsu(image)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)


def resize_for_osd(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    max_dim = max(h, w)

    scale = 1.0
    if max_dim > OSD_MAX_DIM:
        scale = OSD_MAX_DIM / max_dim
    elif max_dim < OSD_MIN_DIM:
        scale = min(OSD_MIN_DIM / max_dim, OSD_MAX_UPSCALE)

    if scale != 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        interpolation = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
        return cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    return image


def resize_for_ocr(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    max_dim = max(h, w)

    if max_dim <= OCR_MAX_DIM:
        return image

    scale = OCR_MAX_DIM / max_dim
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def rotate_image(image: np.ndarray, rotation: int) -> np.ndarray:
    if rotation == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if rotation == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if rotation == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image.copy()


# Pipelines for OSD detection
OSD_PIPELINES = [
    ("grayscale", to_grayscale),
    ("otsu", preprocess_otsu),
    ("clahe", preprocess_clahe),
    ("clahe_otsu", preprocess_clahe_otsu),
    ("adaptive_gaussian", preprocess_adaptive_gaussian),
    ("denoise_otsu", preprocess_denoise_otsu),
    ("sharpen_otsu", preprocess_sharpen_otsu),
    ("morphological", preprocess_morphological),
]

# Pipelines for OCR quality comparison
OCR_PIPELINES = [
    ("grayscale", to_grayscale),
    ("otsu", preprocess_otsu),
    ("clahe_otsu", preprocess_clahe_otsu),
    ("denoise_otsu", preprocess_denoise_otsu),
]
