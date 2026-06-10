"""
visualize_rgb_depth_mask.py
---------------------------
Buat visualisasi poster-quality:
  - Grid RGB | Depth | Mask | Depth-in-Mask
  - Ground-truth vs Prediction overlay
  - Contoh error (false positive, under-segmentation, dll.)

Cara pakai:
    python src/visualize_rgb_depth_mask.py
    python src/visualize_rgb_depth_mask.py --split test --n 6
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

IMG_EXTS = {".jpg", ".jpeg", ".png"}
OUT_DIR  = Path("outputs/poster_figures")


# ── Utilitas ─────────────────────────────────────────────────────────────────

def read_yolo_seg_mask(label_path: Path, img_shape: tuple) -> np.ndarray:
    """Konversi label YOLO-seg polygon ke binary mask gambar penuh."""
    h, w = img_shape[:2]
    combined_mask = np.zeros((h, w), dtype=np.uint8)

    if not label_path.exists():
        return combined_mask

    with open(label_path) as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) < 7:
                continue
            coords = np.array(tokens[1:], dtype=float)
            pts    = coords.reshape(-1, 2)
            pts[:, 0] *= w
            pts[:, 1] *= h
            pts = pts.astype(np.int32)
            cv2.fillPoly(combined_mask, [pts], 1)

    return combined_mask


def load_depth_norm(path: Path, target_shape: tuple = None) -> np.ndarray:
    """Load dan normalisasi depth ke [0,1]."""
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        return None
    depth = depth.astype(np.float32)
    if depth.ndim == 3:
        depth = depth[:, :, 0]

    if target_shape is not None:
        h, w = target_shape[:2]
        if depth.shape[:2] != (h, w):
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)

    dmin, dmax = depth.min(), depth.max()
    if dmax > dmin:
        depth = (depth - dmin) / (dmax - dmin)
    return depth


def apply_colormap(img_gray: np.ndarray, cmap="turbo") -> np.ndarray:
    """Konversi grayscale [0,1] ke BGR colormap."""
    cmap_fn = plt.get_cmap(cmap)
    colored = (cmap_fn(img_gray)[:, :, :3] * 255).astype(np.uint8)
    return cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)


def overlay_mask(rgb: np.ndarray, mask: np.ndarray,
                 color_bgr=(0, 230, 100), alpha=0.45) -> np.ndarray:
    overlay  = rgb.copy()
    colored  = np.zeros_like(rgb)
    colored[mask.astype(bool)] = color_bgr
    blended  = cv2.addWeighted(overlay, 1 - alpha, colored, alpha, 0)
    # Contour tepi mask
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blended, contours, -1, color_bgr, 2)
    return blended


# ── Figure Utama: Grid Poster ─────────────────────────────────────────────────

def make_poster_grid(samples: list, out_path: Path, title: str = ""):
    """
    Setiap sample: dict dengan keys rgb, depth, gt_mask, pred_mask (opsional).
    Buat grid N_row × 4 panel (atau 5 jika ada pred_mask).
    """
    n = len(samples)
    has_pred = any(s.get("pred_mask") is not None for s in samples)
    n_cols   = 5 if has_pred else 4
    col_labels = ["RGB", "Depth Map", "GT Mask", "Depth in Mask"]
    if has_pred:
        col_labels.insert(3, "Pred Mask")

    fig_w = n_cols * 3.2
    fig_h = n * 3.0 + 1.0
    fig, axes = plt.subplots(n, n_cols, figsize=(fig_w, fig_h))
    if n == 1:
        axes = [axes]

    # Header kolom
    for c, lbl in enumerate(col_labels):
        axes[0][c].set_title(lbl, fontsize=11, fontweight="bold", pad=4)

    for row, sample in enumerate(samples):
        rgb       = sample["rgb"]
        depth     = sample["depth"]
        gt_mask   = sample["gt_mask"]
        pred_mask = sample.get("pred_mask")

        rgb_rgb   = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        depth_col = apply_colormap(depth, "turbo") if depth is not None else np.zeros_like(rgb)
        depth_col = cv2.cvtColor(depth_col, cv2.COLOR_BGR2RGB)

        col = 0
        axes[row][col].imshow(rgb_rgb);           col += 1
        axes[row][col].imshow(depth_col);         col += 1
        axes[row][col].imshow(rgb_rgb)
        if gt_mask is not None:
            axes[row][col].imshow(gt_mask, alpha=0.45, cmap="Greens")
        col += 1

        if has_pred:
            axes[row][col].imshow(rgb_rgb)
            if pred_mask is not None:
                axes[row][col].imshow(pred_mask, alpha=0.45, cmap="Oranges")
            col += 1

        # Depth in mask
        if depth is not None and gt_mask is not None:
            masked_d = np.where(gt_mask.astype(bool), depth, np.nan)
            axes[row][col].imshow(masked_d, cmap="plasma",
                                  vmin=np.nanmin(masked_d) if not np.all(np.isnan(masked_d)) else 0,
                                  vmax=np.nanmax(masked_d) if not np.all(np.isnan(masked_d)) else 1)
        else:
            axes[row][col].imshow(np.zeros_like(rgb_rgb))

        axes[row][col].set_title(f"Depth (mask)", fontsize=7)

        # Hapus ticks semua panel
        for ax in axes[row]:
            ax.axis("off")

        # Label baris
        axes[row][0].set_ylabel(sample.get("name", f"Sample {row+1}"),
                                fontsize=8, rotation=0, labelpad=50, va="center")

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Poster grid disimpan: {out_path}")


# ── Figure Depth Profil 3D Surface Plot ───────────────────────────────────────

def make_depth_surface(depth: np.ndarray, mask: np.ndarray, out_path: Path,
                       title: str = "Depth Surface Pothole"):
    """Surface plot 3D area pothole menggunakan matplotlib."""
    ys, xs = np.where(mask.astype(bool))
    if len(xs) == 0:
        return

    # Crop ke bounding box mask + padding
    pad   = 20
    y_min = max(0, ys.min() - pad)
    y_max = min(depth.shape[0], ys.max() + pad)
    x_min = max(0, xs.min() - pad)
    x_max = min(depth.shape[1], xs.max() + pad)

    region_depth = depth[y_min:y_max, x_min:x_max].copy()
    region_mask  = mask[y_min:y_max, x_min:x_max].astype(bool)

    # Mask area di luar pothole dengan mean depth (jangan terlalu jauh)
    fill_val = float(np.mean(region_depth[region_mask])) if region_mask.any() else 0.0
    region_depth_viz = np.where(region_mask, region_depth, fill_val)

    X_grid, Y_grid = np.meshgrid(
        np.arange(region_depth_viz.shape[1]),
        np.arange(region_depth_viz.shape[0])
    )

    # Downsample untuk performa
    step = max(1, region_depth_viz.shape[0] // 80)
    Xd   = X_grid[::step, ::step]
    Yd   = Y_grid[::step, ::step]
    Zd   = region_depth_viz[::step, ::step]

    fig = plt.figure(figsize=(10, 7))
    ax  = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(Xd, Yd, Zd, cmap="plasma", edgecolor="none", alpha=0.85)
    fig.colorbar(surf, ax=ax, shrink=0.5, pad=0.1, label="Nilai Depth (relatif)")
    ax.set_xlabel("X (piksel)")
    ax.set_ylabel("Y (piksel)")
    ax.set_zlabel("Depth")
    ax.set_title(title, fontsize=12)
    ax.view_init(elev=30, azim=-60)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Surface plot 3D disimpan: {out_path}")


# ── Pipeline ─────────────────────────────────────────────────────────────────

def collect_samples(split: str, n: int, base: Path) -> list:
    img_dir   = base / split / "images"
    lbl_dir   = base / split / "labels"
    dep_dir   = base / split / "depth"

    if not img_dir.exists():
        print(f"[WARN] Folder tidak ada: {img_dir}")
        return []

    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    random.shuffle(images)
    images = images[:n]

    samples = []
    for img_path in images:
        stem = img_path.stem
        rgb  = cv2.imread(str(img_path))
        if rgb is None:
            continue

        # Label
        lbl_path = lbl_dir / f"{stem}.txt"
        gt_mask  = read_yolo_seg_mask(lbl_path, rgb.shape)

        # Depth
        depth = None
        for ext in [".png", ".jpg"]:
            dp = dep_dir / f"{stem}{ext}"
            if dp.exists():
                depth = load_depth_norm(dp, rgb.shape)
                break
            dp = dep_dir / f"{stem}_depth{ext}"
            if dp.exists():
                depth = load_depth_norm(dp, rgb.shape)
                break

        samples.append({
            "name"    : stem[:20],
            "rgb"     : rgb,
            "depth"   : depth,
            "gt_mask" : gt_mask,
        })

    return samples


def parse_args():
    p = argparse.ArgumentParser(description="Visualisasi RGB-Depth-Mask untuk poster")
    p.add_argument("--split",   default="test",  choices=["train", "valid", "test"])
    p.add_argument("--n",       type=int, default=6, help="Jumlah sampel")
    p.add_argument("--base",    default="data/pothrgbd")
    p.add_argument("--seed",    type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    print("=" * 60)
    print("  PothRGBD — Visualisasi RGB-Depth-Mask")
    print("=" * 60)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = Path(args.base)

    samples = collect_samples(args.split, args.n, base)
    if not samples:
        print(f"[ERROR] Tidak ada sampel ditemukan di {base / args.split}")
        return

    print(f"[INFO] {len(samples)} sampel dimuat dari split '{args.split}'")

    # 1. Poster grid
    make_poster_grid(
        samples,
        OUT_DIR / f"poster_grid_{args.split}.png",
        title=f"PothRGBD — RGB · Depth · Segmentation Mask ({args.split})"
    )

    # 2. Surface plot untuk sampel pertama yang punya depth + mask
    for s in samples:
        if s["depth"] is not None and s["gt_mask"] is not None and s["gt_mask"].any():
            make_depth_surface(
                s["depth"], s["gt_mask"],
                OUT_DIR / f"depth_surface_{s['name']}.png",
                title=f"Profil Depth Pothole — {s['name']}"
            )
            break

    print(f"\n[DONE] Semua figure tersimpan di: {OUT_DIR}")


if __name__ == "__main__":
    main()
