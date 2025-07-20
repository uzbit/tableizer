#!/usr/bin/env python3
"""
check_labels.py – verifies that every image in images/*/*
has a matching .txt file in labels/*/*  (and vice-versa)
"""

from pathlib import Path
import argparse, textwrap

def collect(root: Path, kind: str):
    return {
        p.relative_to(root / f"{kind}s").with_suffix("")
        for p in (root / f"{kind}s").rglob("*.*")     # jpg, png, txt …
        if p.is_file()
    }

def main():
    ap = argparse.ArgumentParser(
        description="Check YOLO image/label parity",
        formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("dataset_root",
                    help="Folder that contains images/train, labels/train, ...")
    ap.add_argument("--show", action="store_true",
                    help="print every mismatch filename")
    args = ap.parse_args()

    root = Path(args.dataset_root).expanduser()
    bad = False
    for split in ("train", "val", "test"):
        imgs = collect(root / "images", split)
        lbls = collect(root / "labels", split)

        missing_lbl = imgs - lbls      # images without labels
        orphan_lbl  = lbls - imgs      # labels without images

        print(f"\n[{split.upper()}]  images: {len(imgs):5d}   labels: {len(lbls):5d}")
        print(f"          missing labels: {len(missing_lbl):3d}")
        print(f"          orphan  labels: {len(orphan_lbl):3d}")

        if args.show:
            for s in sorted(missing_lbl):
                print("  img→lbl missing:", s)
            for s in sorted(orphan_lbl):
                print("  lbl→img missing:", s)

        bad |= bool(missing_lbl or orphan_lbl)

    if not bad:
        print("\n✅  All images have labels and vice-versa!")

if __name__ == "__main__":
    main()


