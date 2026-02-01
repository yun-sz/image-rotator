import os
from pathlib import Path

import cv2

from image_rotator import process_image

INPUT_DIR = "input"
OUTPUT_DIR = "output"
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = sorted(
        f for f in Path(INPUT_DIR).iterdir()
        if f.is_file() and f.suffix.lower() in EXTENSIONS
    )

    if not files:
        print(f"No images found in {INPUT_DIR}")
        return

    print(f"Found {len(files)} images\n")

    for i, filepath in enumerate(files, 1):
        image = cv2.imread(str(filepath))
        if image is None:
            print(f"[{i}/{len(files)}] SKIP {filepath.name} (failed to load)")
            continue

        rotated, result = process_image(image)

        if result.method == "BlankPage":
            print(f"[{i}/{len(files)}] SKIP {filepath.name} (blank page)")
            continue

        output_path = os.path.join(OUTPUT_DIR, filepath.name)
        cv2.imwrite(output_path, rotated)

        print(f"[{i}/{len(files)}] OK {filepath.name} -> {result.rotation}° ({result.method}) {result.time_ms:.0f}ms")

    print("\nDone.")


if __name__ == "__main__":
    main()
