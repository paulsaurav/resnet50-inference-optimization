"""
Reorganize Kaggle ImageNet val images into ImageFolder layout.

Kaggle ships val images flat:
    imagenet/ILSVRC/Data/CLS-LOC/val/ILSVRC2012_val_00000001.JPEG ...
with labels in LOC_val_solution.csv:
    ImageId,PredictionString
    ILSVRC2012_val_00000001,n01751748 <bbox...> n01751748 <bbox...>

The first WNID in PredictionString is the ground-truth class. We move each
image into val/<wnid>/. After this, datasets.ImageFolder works and — because
all 1000 WNIDs sorted alphabetically match torchvision's expected index order —
the labels line up correctly, so you get the real ~80.3% ResNet-50 top-1.

Usage:
    python src/prepare_imagenet_val.py \
        --val-dir imagenet/ILSVRC/Data/CLS-LOC/val \
        --csv imagenet/LOC_val_solution.csv
"""

import csv
import shutil
import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-dir", required=True, help="flat dir of val JPEGs")
    ap.add_argument("--csv", required=True, help="LOC_val_solution.csv")
    ap.add_argument(
        "--copy", action="store_true",
        help="copy instead of move (keeps originals; needs 2x disk)",
    )
    args = ap.parse_args()

    val_dir = Path(args.val_dir)
    assert val_dir.is_dir(), f"not a dir: {val_dir}"

    # Read image_id -> wnid from the first token of PredictionString.
    mapping = {}
    with open(args.csv, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            if not row:
                continue
            image_id = row[0]
            wnid = row[1].split()[0]
            mapping[image_id] = wnid

    moved = missing = 0
    op = shutil.copy2 if args.copy else shutil.move
    for image_id, wnid in mapping.items():
        src = val_dir / f"{image_id}.JPEG"
        if not src.exists():
            missing += 1
            continue
        dst_dir = val_dir / wnid
        dst_dir.mkdir(exist_ok=True)
        op(str(src), str(dst_dir / src.name))
        moved += 1

    n_classes = len({w for w in mapping.values()})
    print(f"organized {moved} images into {n_classes} WNID folders")
    if missing:
        print(f"warning: {missing} images referenced in CSV were not found")
    print(f"point the validator at: {val_dir}")


if __name__ == "__main__":
    main()