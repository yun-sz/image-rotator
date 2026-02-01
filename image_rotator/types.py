from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class OSDResult:
    rotation: int
    confidence: float
    preprocessing: str


@dataclass
class OrientationResult:
    rotation: int
    confidence: float
    method: str
    votes: Dict[int, int] = field(default_factory=dict)
    details: List[OSDResult] = field(default_factory=list)


@dataclass
class ProcessingResult:
    success: bool
    rotation: int
    confidence: float
    method: str
    time_ms: float
    ocr_text: str = ""
    error: str = ""
