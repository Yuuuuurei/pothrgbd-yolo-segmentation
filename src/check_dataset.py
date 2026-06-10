"""
check_dataset.py
----------------
Periksa kelengkapan dataset PothRGBD:
  - Setiap RGB punya label segmentasi?
  - Setiap RGB punya pasangan depth?
  - Format label YOLO-seg valid?

Cara pakai:
    python src/check_dataset.py
"""

import os
import sys
from pathlib import Path

BASE = Path("data/pothrgbd")
SPLITS = ["train", "valid", "test"]
# IMG_EXTS = {".jpg", ".jpeg", ".png"}
# DEPTH_SUFFIXES = ["", "_depth"]          # stem + suffix + ext
IMG_EXTS = {".jpg", ".jpeg", ".png"}
DEPTH_EXTS = {".npy"}
DEPTH_SUFFIXES = [""]

def find_depth(depth_dir: Path, stem: str) -> Path | None:
    for suffix in DEPTH_SUFFIXES:
        for ext in DEPTH_EXTS:
            p = depth_dir / f"{stem}{suffix}{ext}"
            if p.exists():
                return p
    return None


def check_label_format(label_path: Path) -> tuple[bool, str]:
    """
    Return (valid, reason).
    YOLO-seg format: class_id x1 y1 x2 y2 ... xn yn
    Minimal: class_id + 3 pasangan xy (segitiga) = 7 token.
    """
    with open(label_path) as f:
        lines = [l.strip() for l in f if l.strip()]

    if not lines:
        return False, "file kosong"

    for i, line in enumerate(lines, 1):
        tokens = line.split()
        if len(tokens) < 7:
            return False, f"baris {i}: terlalu sedikit token ({len(tokens)})"
        try:
            int(tokens[0])            # class_id harus integer
            coords = [float(t) for t in tokens[1:]]
        except ValueError:
            return False, f"baris {i}: nilai non-numerik"

        if len(coords) % 2 != 0:
            return False, f"baris {i}: jumlah koordinat ganjil"

        if not all(0.0 <= c <= 1.0 for c in coords):
            return False, f"baris {i}: koordinat di luar [0,1]"

    return True, "ok"


def check_split(split: str) -> dict:
    img_dir   = BASE / split / "images"
    lbl_dir   = BASE / split / "labels"
    dep_dir   = BASE / split / "depth"

    if not img_dir.exists():
        print(f"  [WARN] Folder tidak ditemukan: {img_dir}")
        return {}

    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)

    missing_label   = []
    missing_depth   = []
    invalid_label   = []
    ok_count        = 0

    for img in images:
        stem = img.stem

        # Cek label
        lbl = lbl_dir / f"{stem}.txt"
        has_label = lbl.exists()
        if not has_label:
            missing_label.append(img.name)

        # Cek depth
        dep = find_depth(dep_dir, stem)
        has_depth = dep is not None
        if not has_depth:
            missing_depth.append(img.name)

        # Validasi format label
        if has_label:
            valid, reason = check_label_format(lbl)
            if not valid:
                invalid_label.append((img.name, reason))

        if has_label and has_depth:
            ok_count += 1

    return {
        "split"        : split,
        "total_images" : len(images),
        "ok"           : ok_count,
        "missing_label": missing_label,
        "missing_depth": missing_depth,
        "invalid_label": invalid_label,
    }


def main():
    print("=" * 60)
    print("  PothRGBD — Dataset Integrity Check")
    print("=" * 60)

    all_ok = True
    grand_total = 0
    grand_ok    = 0

    for split in SPLITS:
        print(f"\n▶ Split: {split}")
        result = check_split(split)
        if not result:
            continue

        t  = result["total_images"]
        ok = result["ok"]
        grand_total += t
        grand_ok    += ok

        print(f"  Total images    : {t}")
        print(f"  Lengkap (✓)     : {ok}")

        if result["missing_label"]:
            all_ok = False
            print(f"  ✗ Missing label : {len(result['missing_label'])}")
            for f in result["missing_label"][:5]:
                print(f"      - {f}")
            if len(result["missing_label"]) > 5:
                print(f"      ... dan {len(result['missing_label'])-5} lainnya")

        if result["missing_depth"]:
            all_ok = False
            print(f"  ✗ Missing depth : {len(result['missing_depth'])}")
            for f in result["missing_depth"][:5]:
                print(f"      - {f}")
            if len(result["missing_depth"]) > 5:
                print(f"      ... dan {len(result['missing_depth'])-5} lainnya")

        if result["invalid_label"]:
            all_ok = False
            print(f"  ✗ Invalid label : {len(result['invalid_label'])}")
            for fname, reason in result["invalid_label"][:5]:
                print(f"      - {fname}: {reason}")

        if not result["missing_label"] and not result["missing_depth"] and not result["invalid_label"]:
            print("  ✓ Semua file lengkap dan valid.")

    print("\n" + "=" * 60)
    print(f"  TOTAL: {grand_ok}/{grand_total} gambar lengkap")
    if all_ok:
        print("  STATUS: ✓ Dataset siap digunakan.")
    else:
        print("  STATUS: ✗ Ada file yang hilang atau invalid. Cek log di atas.")
    print("=" * 60)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
