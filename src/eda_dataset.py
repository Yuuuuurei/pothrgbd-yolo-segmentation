"""
eda_dataset.py  (juga tersedia sebagai notebooks/01_dataset_check.ipynb)
------------------------------------------------------------------------
Exploratory Data Analysis PothRGBD.

Analisis:
  1. Jumlah pasangan RGB-depth per split
  2. Jumlah instance mask per gambar
  3. Distribusi area mask relatif (% area gambar)
  4. Distribusi aspek rasio bounding box
  5. Contoh visualisasi RGB-depth-mask

Cara pakai:
    python src/eda_dataset.py
    python src/eda_dataset.py --split train --max 500
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import re

BASE    = Path("data/pothrgbd")
SPLITS  = ["train", "valid", "test"]
IMG_EXTS = {".jpg", ".jpeg", ".png"}
OUT_DIR = Path("outputs/eda")


def polygon_area(coords: np.ndarray) -> float:
    """
    Hitung luas polygon dengan shoelace formula.
    Koordinat diasumsikan sudah ternormalisasi dalam rentang [0, 1].
    Output berupa area relatif terhadap luas gambar.
    """
    x = coords[:, 0]
    y = coords[:, 1]

    return float(
        0.5 * abs(
            np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
        )
    )

def extract_numbers_from_line(line: str) -> list[float]:
    """
    Ekstraksi angka dari baris label.

    Dibuat lebih robust untuk kasus angka yang menempel, misalnya:
        0.73333333333333330.378125

    Regex akan membaca sebagai:
        0.73333333333333330
        .378125
    """
    pattern = r"[-+]?(?:\d*\.\d+|\d+)"
    nums = re.findall(pattern, line)
    return [float(x) for x in nums]

def read_labels(label_path: Path) -> list[dict]:
    """
    Parse YOLO-seg label.

    Format:
        class_id x1 y1 x2 y2 ... xn yn

    Return list of instance dicts.
    """
    instances = []

    if not label_path.exists():
        return instances

    with open(label_path) as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            nums = extract_numbers_from_line(line)

            # Minimal YOLO-seg:
            # class_id + 3 pasangan koordinat = 7 angka
            if len(nums) < 7:
                continue

            cls_id = int(nums[0])
            values = np.array(nums[1:], dtype=float)

            # Koordinat harus berpasangan x,y
            if len(values) % 2 != 0:
                continue

            coords = values.reshape(-1, 2)

            # Abaikan jika ada koordinat di luar rentang normalisasi
            if np.any(coords < 0.0) or np.any(coords > 1.0):
                continue

            xs, ys = coords[:, 0], coords[:, 1]
            bbox_w = xs.max() - xs.min()
            bbox_h = ys.max() - ys.min()
            area = polygon_area(coords)

            instances.append({
                "class_id": cls_id,
                "n_points": len(coords),
                "bbox_w": float(bbox_w),
                "bbox_h": float(bbox_h),
                "bbox_area": float(bbox_w * bbox_h),
                "poly_area": area,
            })

    return instances


def scan_split(split: str, max_samples: int) -> dict:
    img_dir = BASE / split / "images"
    lbl_dir = BASE / split / "labels"
    dep_dir = BASE / split / "depth"

    if not img_dir.exists():
        return {}

    images  = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if max_samples:
        images = images[:max_samples]

    has_depth = 0
    instances_per_img = []
    all_bbox_areas    = []
    all_poly_areas    = []
    all_bbox_aspects  = []

    for img_path in images:
        stem = img_path.stem

        # depth
        # for ext in [".png", ".jpg"]:
        #     dp1 = dep_dir / f"{stem}{ext}"
        #     dp2 = dep_dir / f"{stem}_depth{ext}"
        # depth_candidates = [
        #     dep_dir / f"{stem}.npy",
        #     dep_dir / f"{stem}_depth.npy",
        #     ]
        # if any(p.exists() for p in depth_candidates):
        #     has_depth += 1
        #     break

        depth_candidates = [
            dep_dir / f"{stem}.npy",
            dep_dir / f"{stem}_depth.npy",
        ]

        if any(p.exists() for p in depth_candidates):
            has_depth += 1

        # labels
        lbl_path = lbl_dir / f"{stem}.txt"
        insts    = read_labels(lbl_path)
        instances_per_img.append(len(insts))

        for inst in insts:
            all_bbox_areas.append(inst["bbox_area"])
            all_poly_areas.append(inst["poly_area"])
            if inst["bbox_h"] > 0:
                all_bbox_aspects.append(inst["bbox_w"] / inst["bbox_h"])

    return {
        "split"              : split,
        "n_images"           : len(images),
        "n_with_depth"       : has_depth,
        "instances_per_img"  : instances_per_img,
        "bbox_areas"         : all_bbox_areas,
        "poly_areas"         : all_poly_areas,
        "bbox_aspects"       : all_bbox_aspects,
    }


def plot_eda(all_data: list[dict], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Fig 1: Jumlah instance per gambar ──────────────────────────────────
    fig, axes = plt.subplots(1, len(all_data), figsize=(5 * len(all_data), 4), sharey=False)
    if len(all_data) == 1:
        axes = [axes]
    fig.suptitle("Distribusi Jumlah Instance Pothole per Gambar", fontsize=13)

    for ax, data in zip(axes, all_data):
        vals = data["instances_per_img"]
        if not vals:
            continue
        bins = range(0, max(vals) + 2)
        ax.hist(vals, bins=bins, color="#3070b3", edgecolor="white", rwidth=0.85)
        ax.set_title(f"Split: {data['split']}  (n={data['n_images']})", fontsize=10)
        ax.set_xlabel("Jumlah instance")
        ax.set_ylabel("Frekuensi gambar")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(out_dir / "instances_per_image.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ── Fig 2: Distribusi area bbox ─────────────────────────────────────────
    fig, axes = plt.subplots(1, len(all_data), figsize=(5 * len(all_data), 4))
    if len(all_data) == 1:
        axes = [axes]
    fig.suptitle("Distribusi Area Bounding Box (relatif, 0–1)", fontsize=13)

    for ax, data in zip(axes, all_data):
        areas = data["bbox_areas"]
        if not areas:
            continue
        ax.hist(areas, bins=40, color="#e05a2b", edgecolor="white")
        ax.set_title(f"Split: {data['split']}", fontsize=10)
        ax.set_xlabel("Area bbox (lebar × tinggi, ternormalisasi)")
        ax.set_ylabel("Frekuensi")
        ax.grid(axis="y", alpha=0.3)
        mu = float(np.mean(areas))
        ax.axvline(mu, color="navy", lw=1.5, label=f"Mean: {mu:.4f}")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(str(out_dir / "bbox_area_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ── Fig 3: Ringkasan tabel ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis("off")
    rows = []
    for d in all_data:
        n    = d["n_images"]
        nd   = d["n_with_depth"]
        inst = d["instances_per_img"]
        rows.append([
            d["split"],
            str(n),
            str(nd),
            f"{n - nd}" if n != nd else "0",
            f"{sum(inst)}",
            f"{np.mean(inst):.2f}" if inst else "—",
            f"{max(inst)}" if inst else "—",
        ])
    tbl = ax.table(
        cellText=rows,
        colLabels=["Split", "N Images", "N Depth", "Missing Depth",
                   "Total Instance", "Avg Inst/Img", "Max Inst"],
        cellLoc="center", loc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.8)
    ax.set_title("Ringkasan Dataset PothRGBD", fontsize=13, pad=20)
    plt.tight_layout()
    plt.savefig(str(out_dir / "dataset_summary_table.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[OK] Semua plot EDA tersimpan di: {out_dir}")


def print_summary(all_data: list[dict]):
    print("\n── Ringkasan Dataset ──────────────────────────────────────────")
    print(f"  {'Split':<8} {'N Images':>9} {'N Depth':>9} {'Total Inst':>11} {'Avg/Img':>8}")
    print("  " + "─" * 52)
    for d in all_data:
        inst = d["instances_per_img"]
        print(f"  {d['split']:<8} {d['n_images']:>9} {d['n_with_depth']:>9} "
              f"{sum(inst):>11} {np.mean(inst):>8.2f}" if inst
              else f"  {d['split']:<8} {d['n_images']:>9} {d['n_with_depth']:>9} {'—':>11} {'—':>8}")
    print("  " + "─" * 52)


def parse_args():
    p = argparse.ArgumentParser(description="EDA dataset PothRGBD")
    p.add_argument("--split", default="all", help="all / train / valid / test")
    p.add_argument("--max",   type=int, default=0, help="Batasi jumlah gambar (0=semua)")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  PothRGBD — Exploratory Data Analysis")
    print("=" * 60)

    splits = SPLITS if args.split == "all" else [args.split]

    all_data = []
    for split in splits:
        print(f"[INFO] Scanning split: {split} ...")
        data = scan_split(split, args.max)
        if data:
            all_data.append(data)

    if not all_data:
        print("[ERROR] Tidak ada data ditemukan. Pastikan dataset sudah didownload.")
        return

    print_summary(all_data)
    plot_eda(all_data, OUT_DIR)


if __name__ == "__main__":
    main()
