from __future__ import annotations

from pathlib import Path

from PIL import Image
from torchvision import transforms

VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    train_dir = base_dir / "data" / "clean_train_data"

    if not train_dir.exists():
        raise FileNotFoundError(f"clean_train_data not found at: {train_dir}")

    color_jitter = transforms.ColorJitter(
        brightness=0.4,
        contrast=0.4,
    )
    random_rotation = transforms.RandomRotation(20)

    processed = 0
    skipped = 0
    created = 0

    for class_dir in sorted(p for p in train_dir.iterdir() if p.is_dir()):
        # Snapshot source files first so newly created augmented files
        # are not re-augmented in the same run.
        source_files = [
            p
            for p in sorted(class_dir.iterdir())
            if p.is_file()
            and p.suffix.lower() in VALID_EXTS
            and "_aug_cj" not in p.stem
            and "_aug_rot" not in p.stem
        ]

        for img_path in source_files:
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                skipped += 1
                print(f"[SKIP] Could not open image: {img_path}")
                continue

            cj_img = color_jitter(img)
            rot_img = random_rotation(img)

            cj_path = class_dir / f"{img_path.stem}_aug_cj{img_path.suffix.lower()}"
            rot_path = class_dir / f"{img_path.stem}_aug_rot{img_path.suffix.lower()}"

            cj_img.save(cj_path)
            rot_img.save(rot_path)

            processed += 1
            created += 2

    print(
        f"[DONE] Processed: {processed} source images | "
        f"Created: {created} augmented images | Skipped: {skipped}"
    )


if __name__ == "__main__":
    main()
