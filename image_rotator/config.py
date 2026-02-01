import os
import re

# Tesseract configs
OSD_CONFIG = "--psm 0"
OCR_CONFIG = "--psm 6"

# Image resize targets
OSD_MAX_DIM = 2000
OSD_MIN_DIM = 1100
OCR_MAX_DIM = 1500
OSD_MAX_UPSCALE = 1.5

# OSD early-exit thresholds
OSD_EARLY_STOP_MIN_RESULTS = 4
OSD_EARLY_STOP_CONF = 35.0

# OSD trust thresholds
OSD_TRUST_MIN_RESULTS = 5
OSD_TRUST_AGREEMENT = 0.85
OSD_TRUST_OSD_CONF = 25.0
OSD_LOW_CONFIDENCE = 12.0

# Minimum OCR score difference ratio to override OSD
MIN_CONFIDENCE_RATIO = 0.10

# Blank page detection threshold
BLANK_PAGE_THRESHOLD = 0.02

# Pipeline weights for voting
PIPELINE_WEIGHTS = {
    "otsu": 1.2,
    "clahe_otsu": 1.2,
    "denoise_otsu": 1.1,
    "grayscale": 1.0,
    "clahe": 1.0,
    "adaptive_gaussian": 0.9,
    "sharpen_otsu": 0.9,
    "morphological": 0.8,
}

# Words that look similar upside down
SYMMETRIC_WORDS = {'MOM', 'WOW', 'SOS', 'OHO', 'XIX'}

# Patterns for text validation
NUMBER_PATTERN = re.compile(r'\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?')
DATE_PATTERN = re.compile(r'\d{2}[/.-]\d{2}[/.-]?\d{0,4}')


def load_common_words(filepath: str = None) -> set:
    if filepath is None:
        filepath = os.path.join(os.path.dirname(__file__), 'palavras.txt')

    words = set()
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip()
                if len(word) > 2 and word.isalpha():
                    words.add(word.upper())
    except FileNotFoundError:
        words = {'TOTAL', 'VALOR', 'DATA', 'CONTA', 'BANCO', 'PAGAMENTO'}

    return words


COMMON_WORDS = load_common_words()
