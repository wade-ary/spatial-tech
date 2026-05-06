from __future__ import annotations

from pathlib import Path

import cv2

TARGET_SIZE = (128, 128)
VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def resize_dataset(src_root: Path, dst_root: Path) -> tuple[int, int]:
    """
    Resize all images in src_root (class subfolders) to TARGET_SIZE and save to dst_root.
    Returns (processed_count, skipped_count).
    """
    if not src_root.exists():
        print(f"[WARN] Source folder not found: {src_root}")
        return 0, 0

    processed = 0
    skipped = 0

    for cls_dir in sorted(p for p in src_root.iterdir() if p.is_dir()):
        out_cls_dir = dst_root / cls_dir.name
        out_cls_dir.mkdir(parents=True, exist_ok=True)

        for img_path in sorted(cls_dir.iterdir()):
            if not img_path.is_file() or img_path.suffix.lower() not in VALID_EXTS:
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                skipped += 1
                print(f"[SKIP] Could not read image: {img_path}")
                continue

            resized = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
            out_path = out_cls_dir / img_path.name
            ok = cv2.imwrite(str(out_path), resized)
            if not ok:
                skipped += 1
                print(f"[SKIP] Could not write image: {out_path}")
                continue

            processed += 1

    return processed, skipped


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"

    jobs = [
        (data_dir / "raw_train_data", data_dir / "clean_train_data"),
        (data_dir / "raw_test_data", data_dir / "clean_test_data"),
    ]

    total_processed = 0
    total_skipped = 0

    for src, dst in jobs:
        dst.mkdir(parents=True, exist_ok=True)
        processed, skipped = resize_dataset(src, dst)
        total_processed += processed
        total_skipped += skipped
        print(
            f"[DONE] {src.name} -> {dst.name} | processed: {processed} | skipped: {skipped}"
        )

    print(
        f"[SUMMARY] target_size={TARGET_SIZE[0]}x{TARGET_SIZE[1]} | "
        f"processed={total_processed} | skipped={total_skipped}"
    )


if __name__ == "__main__":
    main()
