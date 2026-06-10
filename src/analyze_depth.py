"""
analyze_depth.py
----------------
Analisis depth pada area pothole yang tersegmentasi.

Fitur:
  - Statistik depth (mean, median, min, max, range, std)
  - Heatmap depth pada area mask
  - Profil depth cross-section (horizontal & vertikal)
  - Point cloud 3D via Open3D (opsional)
  - Export ringkasan ke CSV

Cara pakai:
    python src/analyze_depth.py
    python src/analyze_depth.py --rgb path.jpg --depth depth.npy --mask pred_mask.npy
    python src/analyze_depth.py --use-pred-dir runs/segment/pothrgbd_predictions/
"""

import argparse
import json
import csv
import warnings
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm

warnings.filterwarnings("ignore")

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("[INFO] open3d tidak terinstall. Fitur point cloud 3D dinonaktifkan.")
    print("       Install: pip install open3d")


IMG_EXTS = {".jpg", ".jpeg", ".png"}
OUT_DIR  = Path("outputs")


# ── Utilitas Depth ───────────────────────────────────────────────────────────

def load_depth(path: Path) -> np.ndarray:
    """
    Load depth sebagai float32.

    Mendukung:
      - .npy depth array
      - .png/.jpg/.jpeg depth image
    """
    if not path.exists():
        raise FileNotFoundError(f"Depth file tidak ditemukan: {path}")

    if path.suffix.lower() == ".npy":
        depth = np.load(path)
    else:
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(f"Depth tidak bisa dibuka: {path}")

    depth = depth.astype(np.float32)

    if depth.ndim == 3:
        depth = depth[:, :, 0]

    return depth


def align_depth_to_rgb(depth: np.ndarray, rgb_shape: tuple) -> np.ndarray:
    """Resize depth ke ukuran RGB jika berbeda."""
    h_rgb, w_rgb = rgb_shape[:2]
    if depth.shape[:2] != (h_rgb, w_rgb):
        depth = cv2.resize(depth, (w_rgb, h_rgb), interpolation=cv2.INTER_NEAREST)
    return depth


def compute_depth_stats(depth_vals: np.ndarray) -> dict:
    """Hitung statistik depth pada area pothole."""
    if depth_vals.size == 0:
        return {}
    return {
        "n_pixels"   : int(depth_vals.size),
        "mean_depth" : float(np.mean(depth_vals)),
        "median_depth": float(np.median(depth_vals)),
        "std_depth"  : float(np.std(depth_vals)),
        "min_depth"  : float(np.min(depth_vals)),
        "max_depth"  : float(np.max(depth_vals)),
        "depth_range": float(np.max(depth_vals) - np.min(depth_vals)),
        "q25_depth"  : float(np.percentile(depth_vals, 25)),
        "q75_depth"  : float(np.percentile(depth_vals, 75)),
    }


# ── Visualisasi ──────────────────────────────────────────────────────────────

def plot_depth_analysis(rgb: np.ndarray, depth: np.ndarray,
                        mask: np.ndarray, stats: dict,
                        out_path: Path, title: str = ""):
    """
    Buat figure 4-panel:
      [RGB] [Depth full] [Mask overlay] [Depth in mask]
    + histogram distribusi depth pothole.
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(title or "Analisis Depth Pothole", fontsize=14, fontweight="bold")

    # Panel 1: RGB
    ax = axes[0, 0]
    ax.imshow(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
    ax.set_title("RGB Image", fontsize=10)
    ax.axis("off")

    # Panel 2: Full depth map
    ax = axes[0, 1]
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    im = ax.imshow(depth_norm, cmap="turbo")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Depth Map (full)", fontsize=10)
    ax.axis("off")

    # Panel 3: Mask overlay on RGB
    ax = axes[0, 2]
    rgb_copy = cv2.cvtColor(rgb.copy(), cv2.COLOR_BGR2RGB)
    overlay  = rgb_copy.copy()
    overlay[mask.astype(bool)] = [0, 200, 100]
    blended  = cv2.addWeighted(rgb_copy, 0.6, overlay, 0.4, 0)
    ax.imshow(blended)
    ax.set_title("Predicted Mask Overlay", fontsize=10)
    ax.axis("off")

    # Panel 4: Depth in mask area
    ax = axes[1, 0]
    masked_depth = np.where(mask.astype(bool), depth, np.nan)
    valid_min    = np.nanmin(masked_depth) if not np.all(np.isnan(masked_depth)) else 0
    valid_max    = np.nanmax(masked_depth) if not np.all(np.isnan(masked_depth)) else 1
    im2 = ax.imshow(masked_depth, cmap="plasma", vmin=valid_min, vmax=valid_max)
    plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Depth di Area Pothole", fontsize=10)
    ax.axis("off")

    # Panel 5: Histogram depth pothole
    ax = axes[1, 1]
    depth_vals = depth[mask.astype(bool)]
    if depth_vals.size > 0:
        ax.hist(depth_vals, bins=50, color="#e05a2b", edgecolor="white", linewidth=0.3)
        ax.axvline(stats.get("mean_depth", 0),   color="blue",  lw=1.5, label=f"Mean: {stats['mean_depth']:.1f}")
        ax.axvline(stats.get("median_depth", 0), color="green", lw=1.5, ls="--", label=f"Median: {stats['median_depth']:.1f}")
        ax.set_xlabel("Nilai Depth (relatif)")
        ax.set_ylabel("Frekuensi Piksel")
        ax.set_title("Distribusi Depth Pothole", fontsize=10)
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "Tidak ada mask", ha="center", va="center")
        ax.axis("off")

    # Panel 6: Statistik teks
    ax = axes[1, 2]
    ax.axis("off")
    if stats:
        rows = [
            ("Jumlah piksel",  f"{stats.get('n_pixels', 0):,}"),
            ("Mean depth",     f"{stats.get('mean_depth', 0):.2f}"),
            ("Median depth",   f"{stats.get('median_depth', 0):.2f}"),
            ("Std depth",      f"{stats.get('std_depth', 0):.2f}"),
            ("Min depth",      f"{stats.get('min_depth', 0):.2f}"),
            ("Max depth",      f"{stats.get('max_depth', 0):.2f}"),
            ("Depth range",    f"{stats.get('depth_range', 0):.2f}"),
            ("Q25 depth",      f"{stats.get('q25_depth', 0):.2f}"),
            ("Q75 depth",      f"{stats.get('q75_depth', 0):.2f}"),
        ]
        table_data = [[r[0], r[1]] for r in rows]
        tbl = ax.table(cellText=table_data, colLabels=["Statistik", "Nilai"],
                       cellLoc="left", loc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1, 1.6)
    ax.set_title("Ringkasan Statistik", fontsize=10)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_depth_profile(depth: np.ndarray, mask: np.ndarray,
                       out_path: Path, title: str = ""):
    """
    Profil depth cross-section melewati centroid mask:
    - Horizontal slice
    - Vertikal slice
    """
    ys, xs = np.where(mask.astype(bool))
    if len(xs) == 0:
        return

    cy = int(np.mean(ys))
    cx = int(np.mean(xs))

    h_profile = depth[cy, :]     # horizontal
    v_profile = depth[:, cx]     # vertikal
    mask_h    = mask[cy, :].astype(bool)
    mask_v    = mask[:, cx].astype(bool)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle(title or "Profil Depth Cross-Section", fontsize=13)

    for ax, profile, mask_1d, label, axis_label in zip(
        axes,
        [h_profile, v_profile],
        [mask_h, mask_v],
        ["Horizontal (baris centroid)", "Vertikal (kolom centroid)"],
        ["Kolom piksel", "Baris piksel"]
    ):
        x_all = np.arange(len(profile))
        ax.plot(x_all, profile, color="gray", lw=0.8, label="Full profile")
        ax.fill_between(x_all, profile, where=mask_1d,
                        alpha=0.5, color="#e05a2b", label="Area pothole")
        ax.set_xlabel(axis_label)
        ax.set_ylabel("Nilai Depth (relatif)")
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Open3D Point Cloud ────────────────────────────────────────────────────────

def create_point_cloud(rgb: np.ndarray, depth: np.ndarray, mask: np.ndarray,
                       out_path: Path, fx: float = 525.0, fy: float = 525.0):
    """
    Buat point cloud dari depth map pada area pothole (Open3D).
    fx, fy: focal length piksel (default estimasi kamera umum).
    """
    if not HAS_OPEN3D:
        return

    h, w = depth.shape
    cx_cam, cy_cam = w / 2, h / 2

    rows, cols = np.where(mask.astype(bool))
    if rows.size == 0:
        print("  [WARN] Mask kosong. Point cloud tidak dibuat.")
        return

    Z = depth[rows, cols].astype(np.float64)

    # Filter depth tidak valid
    valid = np.isfinite(Z) & (Z > 0)
    rows = rows[valid]
    cols = cols[valid]
    Z = Z[valid]

    if Z.size == 0:
        print("  [WARN] Tidak ada nilai depth valid. Point cloud tidak dibuat.")
        return

    X = (cols - cx_cam) * Z / fx
    Y = (rows - cy_cam) * Z / fy

    points = np.stack([X, Y, Z], axis=1)

    # Warna dari RGB; rows dan cols sudah difilter, jadi jangan difilter lagi
    rgb_uint8 = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    colors = rgb_uint8[rows, cols] / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if colors.shape[0] == points.shape[0]:
        pcd.colors = o3d.utility.Vector3dVector(colors)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(out_path), pcd)
    print(f"  [OK] Point cloud disimpan: {out_path}  ({len(points)} poin)")

    print("  [INFO] Render otomatis dilewati. Buka file .ply dengan MeshLab, CloudCompare, atau Open3D viewer.")
    # try:
    #     render = o3d.visualization.rendering.OffscreenRenderer(800, 600)
    #     mat = o3d.visualization.rendering.MaterialRecord()
    #     mat.shader = "defaultUnlit"
    #     mat.point_size = 3.0

    #     render.scene.add_geometry("pcd", pcd, mat)
    #     render.scene.set_background([0.1, 0.1, 0.1, 1.0])

    #     bounds = pcd.get_axis_aligned_bounding_box()
    #     center = bounds.get_center()
    #     extent = bounds.get_max_extent()

    #     if extent > 0:
    #         render.setup_camera(
    #             60.0,
    #             center,
    #             center + np.array([0, 0, -extent]),
    #             [0, -1, 0],
    #         )

    #         img_o3d = render.render_to_image()
    #         o3d.io.write_image(str(out_path.with_suffix(".png")), img_o3d)
    #         print(f"  [OK] Render point cloud disimpan: {out_path.with_suffix('.png')}")

    # except Exception as e:
    #     print(f"  [WARN] Render headless gagal ({e}). PCD file tetap tersimpan.")


# ── Pipeline Utama ────────────────────────────────────────────────────────────

def run_single(rgb_path: Path, depth_path: Path, mask: np.ndarray,
               prefix: str = "sample", save_pcd: bool = True):
    """
    Analisis satu pasang RGB-depth dengan mask prediksi.
    Return stats dict.
    """
    rgb   = cv2.imread(str(rgb_path))
    depth = load_depth(depth_path)
    depth = align_depth_to_rgb(depth, rgb.shape)

    # Pastikan mask binary
    if mask.ndim == 3:
        mask = mask.squeeze()
    bin_mask = (mask > 0.5).astype(np.uint8)

    depth_vals = depth[bin_mask.astype(bool)]
    stats      = compute_depth_stats(depth_vals)

    # Simpan visualisasi analisis
    vis_path = OUT_DIR / "depth_profiles" / f"{prefix}_analysis.png"
    plot_depth_analysis(rgb, depth, bin_mask, stats, vis_path, title=prefix)

    # Profil cross-section
    prof_path = OUT_DIR / "depth_profiles" / f"{prefix}_profile.png"
    plot_depth_profile(depth, bin_mask, prof_path, title=prefix)

    # Point cloud
    if save_pcd and HAS_OPEN3D:
        pcd_path = OUT_DIR / "depth_profiles" / f"{prefix}_pointcloud.ply"
        create_point_cloud(rgb, depth, bin_mask, pcd_path)

    return stats


def run_batch_from_predictions(pred_dir: Path, depth_base: Path,
                               max_samples: int = 20):
    """
    Loop prediksi YOLO, pasangkan dengan depth, analisis setiap sampel.
    pred_dir : folder hasil model.predict(..., save=True) yang berisi gambar hasil.
    depth_base: folder depth test (e.g. data/pothrgbd/test/depth/)
    """
    print(f"\n[INFO] Batch depth analysis dari: {pred_dir}")

    # Cari gambar prediksi
    img_files = sorted(pred_dir.glob("*.jpg")) + sorted(pred_dir.glob("*.png"))
    img_files = img_files[:max_samples]

    all_stats = []
    for img_path in tqdm(img_files, desc="Analisis depth"):
        stem = img_path.stem

        # Cari depth pair
        dep_path = None
        for ext in [".npy", ".png", ".jpg", ".jpeg"]:
            p = depth_base / f"{stem}{ext}"
            if p.exists():
                dep_path = p
                break

            p = depth_base / f"{stem}_depth{ext}"
            if p.exists():
                dep_path = p
                break

        if dep_path is None:
            continue

        # Buat dummy mask (seluruh gambar) jika tidak ada mask terpisah
        # Dalam penggunaan nyata, mask diambil dari r.masks
        rgb = cv2.imread(str(img_path))
        dummy_mask = np.ones(rgb.shape[:2], dtype=np.uint8)

        try:
            stats = run_single(img_path, dep_path, dummy_mask, prefix=stem, save_pcd=False)
            stats["file"] = stem
            all_stats.append(stats)
        except Exception as e:
            print(f"  [WARN] Gagal analisis {stem}: {e}")

    # Simpan rangkuman CSV
    if all_stats:
        csv_path = OUT_DIR / "depth_analysis_summary.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_stats[0].keys())
            writer.writeheader()
            writer.writerows(all_stats)
        print(f"\n[OK] Ringkasan CSV disimpan: {csv_path}")

    return all_stats


# ── Main CLI ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Analisis depth pada area pothole")
    p.add_argument("--rgb",      default=None, help="Path gambar RGB")
    p.add_argument("--depth",    default=None, help="Path depth image")
    p.add_argument("--mask",     default=None, help="Path mask .npy (bool array H×W)")
    p.add_argument("--pred-dir", default=None,
                   help="Folder hasil model.predict — mode batch")
    p.add_argument("--depth-dir", default="data/pothrgbd/test/depth",
                   help="Folder depth untuk mode batch")
    p.add_argument("--max",      type=int, default=20,
                   help="Maks sampel pada mode batch")
    p.add_argument("--no-pcd",   action="store_true",
                   help="Skip pembuatan point cloud")
    return p.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print("  PothRGBD — Depth Analysis")
    print("=" * 60)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "depth_profiles").mkdir(parents=True, exist_ok=True)

    if args.pred_dir:
        run_batch_from_predictions(
            Path(args.pred_dir),
            Path(args.depth_dir),
            max_samples=args.max
        )
    elif args.rgb and args.depth:
        rgb_path   = Path(args.rgb)
        depth_path = Path(args.depth)

        if args.mask:
            mask = np.load(args.mask)
        else:
            # Mask seluruh gambar sebagai placeholder
            rgb = cv2.imread(str(rgb_path))
            mask = np.ones(rgb.shape[:2], dtype=np.uint8)
            print("[INFO] Mask tidak disediakan. Menganalisis seluruh area gambar.")

        stats = run_single(rgb_path, depth_path, mask,
                           prefix=rgb_path.stem,
                           save_pcd=not args.no_pcd)

        print("\n── Statistik Depth ──────────────────────────────────")
        for k, v in stats.items():
            print(f"  {k:<15}: {v}")
        print("─────────────────────────────────────────────────────")
    else:
        print("[INFO] Gunakan --rgb dan --depth untuk single image,")
        print("       atau --pred-dir untuk batch mode.")
        print("\nContoh:")
        print("  python src/analyze_depth.py --rgb img.jpg --depth dep.npy")
        print("  python src/analyze_depth.py --pred-dir runs/segment/pothrgbd_predictions/")

    print("\n[DONE] Analisis selesai. Output di: outputs/depth_profiles/")


if __name__ == "__main__":
    main()
