import os
from pathlib import Path
from multiprocessing import Pool, cpu_count
from dataclasses import dataclass
from typing import Tuple, Optional

import cv2
import numpy as np

from image_rotator import process_image

INPUT_DIR = "input"
OUTPUT_DIR = "output"
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
NUM_WORKERS = 4


@dataclass
class TaskResult:
    filepath: Path
    success: bool
    rotation: int = 0
    method: str = ""
    time_ms: float = 0
    error: str = ""


def process_single_image(filepath: Path) -> TaskResult:
    try:
        image = cv2.imread(str(filepath))
        if image is None:
            return TaskResult(filepath, success=False, error="failed to load")

        rotated, result = process_image(image)

        if result.method == "BlankPage":
            return TaskResult(filepath, success=False, error="blank page")

        output_path = os.path.join(OUTPUT_DIR, filepath.name)
        cv2.imwrite(output_path, rotated)

        return TaskResult(
            filepath=filepath,
            success=True,
            rotation=result.rotation,
            method=result.method,
            time_ms=result.time_ms
        )

    except Exception as e:
        return TaskResult(filepath, success=False, error=str(e))


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = sorted(
        f for f in Path(INPUT_DIR).iterdir()
        if f.is_file() and f.suffix.lower() in EXTENSIONS
    )

    if not files:
        print(f"No images found in {INPUT_DIR}")
        return

    print(f"Found {len(files)} images")
    print(f"Using {NUM_WORKERS} parallel workers\n")

    with Pool(processes=NUM_WORKERS) as pool:
        results = pool.map(process_single_image, files)

    ok_count = 0
    skip_count = 0
    total_time = 0

    for i, result in enumerate(results, 1):
        if result.success:
            ok_count += 1
            total_time += result.time_ms
            print(f"[{i}/{len(files)}] OK {result.filepath.name} -> {result.rotation}deg ({result.method}) {result.time_ms:.0f}ms")
        else:
            skip_count += 1
            print(f"[{i}/{len(files)}] SKIP {result.filepath.name} ({result.error})")

    print(f"\nDone: {ok_count} processed, {skip_count} skipped")
    if ok_count > 0:
        print(f"Average time per image: {total_time/ok_count:.0f}ms")


if __name__ == "__main__":
    main()
