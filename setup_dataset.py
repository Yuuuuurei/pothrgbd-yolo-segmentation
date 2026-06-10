"""
setup_dataset.py
----------------
Download PothRGBD dataset dari Kaggle dan atur ke struktur folder project.

Prasyarat:
    pip install kaggle
    Letakkan kaggle.json di ~/.kaggle/kaggle.json
    atau set env var KAGGLE_USERNAME dan KAGGLE_KEY

Cara pakai:
    python setup_dataset.py
    python setup_dataset.py --force   # paksa re-download meski folder sudah ada
"""

import os
import sys
import shutil
import argparse
import zipfile
from pathlib import Path
import random
import re

# ── Konfigurasi ──────────────────────────────────────────────────────────────
KAGGLE_DATASET = "mahyeks/pothrgbd-rgb-and-depth-images-of-potholes"   # ganti dengan slug dataset yang benar
BASE_DATA_DIR  = Path("data/pothrgbd")
SPLITS         = ["train", "valid", "test"]
SUBDIRS        = ["images", "labels", "depth"]
DOWNLOAD_DIR   = Path("data/_raw")
TRAIN_RATIO    = 0.70
VALID_RATIO    = 0.20
TEST_RATIO     = 0.10
RANDOM_SEED    = 42
# ─────────────────────────────────────────────────────────────────────────────


def check_kaggle_credentials() -> bool:
    """Periksa apakah kredensial Kaggle tersedia."""
    kaggle_json = Path.home() / ".kaggle" / "access_token"
    env_ok = os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")
    if kaggle_json.exists() or env_ok:
        return True
    print(
        "[ERROR] Kaggle credentials tidak ditemukan.\n"
        "  Opsi 1: Letakkan kaggle.json di ~/.kaggle/kaggle.json\n"
        "  Opsi 2: Set env var KAGGLE_USERNAME dan KAGGLE_KEY\n"
        "  Download kaggle.json dari: https://www.kaggle.com/settings → API → Create New Token"
    )
    return False


def folder_is_populated(path: Path) -> bool:
    """Return True jika folder ada dan memiliki setidaknya satu file."""
    if not path.exists():
        return False
    files = list(path.rglob("*"))
    return any(f.is_file() for f in files)


def ensure_structure():
    """Buat struktur folder project jika belum ada."""
    for split in SPLITS:
        for sub in SUBDIRS:
            d = BASE_DATA_DIR / split / sub
            d.mkdir(parents=True, exist_ok=True)
    print("[OK] Struktur folder data/ sudah siap.")


def download_dataset():
    """Download dataset dari Kaggle ke DOWNLOAD_DIR."""
    try:
        import kaggle  # noqa: F401 — just to verify install
    except ImportError:
        print("[ERROR] Library 'kaggle' belum terinstall. Jalankan: pip install kaggle")
        sys.exit(1)

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Mendownload dataset '{KAGGLE_DATASET}' ke {DOWNLOAD_DIR} ...")

    import subprocess
    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET,
         "-p", str(DOWNLOAD_DIR), "--unzip"],
        capture_output=False
    )
    if result.returncode != 0:
        print("[ERROR] Download gagal. Pastikan slug dataset benar dan kredensial valid.")
        sys.exit(1)
    print("[OK] Download selesai.")

def find_pothrgbd_root() -> Path:
    """
    Cari root dataset PothRGBD di dalam DOWNLOAD_DIR.
    Struktur yang ditemukan pada Kaggle:
        PUBLIC POTHOLE DATASET/
        ├─ images/
        ├─ labels/
        └─ depths/
    """
    candidates = []

    for p in DOWNLOAD_DIR.rglob("*"):
        if p.is_dir():
            has_images = (p / "images").exists()
            has_labels = (p / "labels").exists()
            has_depths = (p / "depths").exists() or (p / "depth").exists()

            if has_images and has_labels and has_depths:
                candidates.append(p)

    if not candidates:
        print("[ERROR] Root dataset tidak ditemukan.")
        print("        Diharapkan ada folder berisi images/, labels/, dan depths/ atau depth/.")
        sys.exit(1)

    root = candidates[0]
    print(f"[OK] Root dataset ditemukan: {root}")
    return root


def get_stem_from_image(path: Path) -> str:
    """Ambil stem dasar dari file image."""
    return path.stem


def find_matching_label(stem: str, labels_dir: Path) -> Path | None:
    """
    Cari file label YOLO yang sesuai dengan image.
    """
    candidates = [
        labels_dir / f"{stem}.txt",
    ]

    for c in candidates:
        if c.exists():
            return c

    return None

def normalize_stem(stem: str) -> str:
    """
    Normalisasi stem agar image/label Roboflow dapat dicocokkan dengan depth asli.

    Contoh:
        20250227_135438_color_png.rf.984e9768... -> 20250227_135438
        20250227_135438_depth                    -> 20250227_135438
    """
    match = re.match(r"^(\d{8}_\d{6})", stem)
    if match:
        return match.group(1)

    suffixes = [
        "_rgb",
        "_RGB",
        "_image",
        "_img",
        "_color",
        "_color_png",
        "_depth",
        "_label",
    ]

    normalized = stem
    for suffix in suffixes:
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]

    return normalized

def find_matching_depth(stem: str, depths_dir: Path) -> Path | None:
    """
    Cari file depth yang sesuai dengan image.
    Image/label memakai nama Roboflow, sedangkan depth memakai nama timestamp asli.
    """
    base = normalize_stem(stem)

    candidates = [
        depths_dir / f"{base}_depth.npy",
        depths_dir / f"{base}.npy",
        depths_dir / f"{stem}_depth.npy",
        depths_dir / f"{stem}.npy",
    ]

    for c in candidates:
        if c.exists():
            return c

    return None


def collect_samples() -> list[dict]:
    """
    Kumpulkan pasangan image-label-depth.
    Hanya sample lengkap yang dipakai.
    """
    root = find_pothrgbd_root()

    images_dir = root / "images"
    labels_dir = root / "labels"
    depths_dir = root / "depths"
    if not depths_dir.exists():
        depths_dir = root / "depth"

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    samples = []
    missing_label = 0
    missing_depth = 0
    empty_label = 0

    for img_path in sorted(images_dir.rglob("*")):
        if not img_path.is_file():
            continue
        if img_path.suffix.lower() not in image_exts:
            continue

        stem = get_stem_from_image(img_path)
        label_path = find_matching_label(stem, labels_dir)
        depth_path = find_matching_depth(stem, depths_dir)

        if label_path is None:
            missing_label += 1
            continue

        if label_path.stat().st_size == 0:
            empty_label += 1
            continue

        if depth_path is None:
            missing_depth += 1
            continue

        samples.append({
            "stem": stem,
            "image": img_path,
            "label": label_path,
            "depth": depth_path,
        })

    print(f"[INFO] Sample lengkap ditemukan : {len(samples)}")
    print(f"[INFO] Image tanpa label       : {missing_label}")
    print(f"[INFO] Image dengan label kosong: {empty_label}")
    print(f"[INFO] Image tanpa depth       : {missing_depth}")

    if len(samples) == 0:
        print("[ERROR] Tidak ada sample lengkap image-label-depth.")
        sys.exit(1)

    return samples


def clear_existing_dataset():
    """Kosongkan folder train/valid/test agar hasil split tidak tercampur."""
    for split in SPLITS:
        split_dir = BASE_DATA_DIR / split
        if split_dir.exists():
            shutil.rmtree(split_dir)

    ensure_structure()


def split_samples(samples: list[dict]) -> dict:
    """Bagi sample menjadi train/valid/test."""
    random.seed(RANDOM_SEED)
    random.shuffle(samples)

    n = len(samples)
    n_train = int(n * TRAIN_RATIO)
    n_valid = int(n * VALID_RATIO)

    train_samples = samples[:n_train]
    valid_samples = samples[n_train:n_train + n_valid]
    test_samples = samples[n_train + n_valid:]

    return {
        "train": train_samples,
        "valid": valid_samples,
        "test": test_samples,
    }


def copy_sample(sample: dict, split: str):
    """Copy satu sample ke folder split tujuan.

    Image dan label memakai nama Roboflow.
    Depth asli memakai nama timestamp, sehingga saat disalin depth
    dibuat mengikuti stem image agar pairing konsisten.
    """
    image_stem = sample["image"].stem

    image_dst = BASE_DATA_DIR / split / "images" / sample["image"].name
    label_dst = BASE_DATA_DIR / split / "labels" / f"{image_stem}.txt"
    depth_dst = BASE_DATA_DIR / split / "depth" / f"{image_stem}.npy"

    shutil.copy2(sample["image"], image_dst)
    shutil.copy2(sample["label"], label_dst)
    shutil.copy2(sample["depth"], depth_dst)


def move_to_structure():
    """
    Susun dataset raw menjadi format project train/valid/test.
    """
    print("[INFO] Menyusun data ke struktur train/valid/test ...")

    samples = collect_samples()
    split_map = split_samples(samples)

    clear_existing_dataset()

    for split, split_samples_ in split_map.items():
        for sample in split_samples_:
            copy_sample(sample, split)

        print(f"  [OK] {split:5s}: {len(split_samples_)} sample")

    print("[OK] Dataset berhasil disusun.")

def copy_files(src: Path, dst: Path, desc: str):
    """Copy semua file dari src ke dst."""
    if src is None or not src.exists():
        print(f"  [SKIP] {desc}: folder sumber tidak ditemukan ({src})")
        return 0
    dst.mkdir(parents=True, exist_ok=True)
    count = 0
    for f in src.iterdir():
        if f.is_file():
            shutil.copy2(f, dst / f.name)
            count += 1
    print(f"  [OK] {desc}: {count} file dipindahkan ke {dst}")
    return count

def create_data_yaml():
    """Buat data.yaml jika belum ada."""
    yaml_path = BASE_DATA_DIR / "data.yaml"
    if yaml_path.exists():
        print(f"[SKIP] {yaml_path} sudah ada.")
        return
    content = (
        f"path: {BASE_DATA_DIR.as_posix()}\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n\n"
        "nc: 1\n"
        "names:\n"
        "  0: pothole\n"
    )
    yaml_path.write_text(content)
    print(f"[OK] {yaml_path} dibuat.")


def print_summary():
    """Tampilkan ringkasan isi folder."""
    print("\n── Ringkasan Dataset ──────────────────────────────")
    for split in SPLITS:
        for sub in SUBDIRS:
            d = BASE_DATA_DIR / split / sub
            n = len(list(d.glob("*"))) if d.exists() else 0
            status = "✓" if n > 0 else "✗ (kosong)"
            print(f"  {split:6s}/{sub:8s}: {n:4d} file {status}")
    print("────────────────────────────────────────────────────\n")


def main():
    parser = argparse.ArgumentParser(description="Setup dataset PothRGBD")
    parser.add_argument("--force", action="store_true",
                        help="Paksa re-download meski data sudah ada")
    parser.add_argument("--slug", default=None,
                        help="Override Kaggle dataset slug (user/dataset-name)")
    args = parser.parse_args()

    global KAGGLE_DATASET
    if args.slug:
        KAGGLE_DATASET = args.slug

    print("=" * 55)
    print("  PothRGBD — Dataset Setup Script")
    print("=" * 55)

    # 1. Buat struktur folder
    ensure_structure()

    # 2. Cek apakah data sudah ada
    train_images = BASE_DATA_DIR / "train" / "images"
    already_populated = folder_is_populated(train_images)

    if already_populated and not args.force:
        print(f"[INFO] Dataset sudah ada di {BASE_DATA_DIR}. Skip download.")
        print("       Gunakan --force untuk re-download.")
    else:
        # 3. Cek kredensial
        if not check_kaggle_credentials():
            sys.exit(1)

        # 4. Download
        download_dataset()

        # 5. Deteksi dan pindahkan
        move_to_structure()

        # 6. Cleanup raw
        if DOWNLOAD_DIR.exists():
            shutil.rmtree(DOWNLOAD_DIR)
            print(f"[OK] Folder raw {DOWNLOAD_DIR} dibersihkan.")

    # 7. Buat data.yaml
    create_data_yaml()

    # 8. Ringkasan
    print_summary()
    print("[DONE] Setup selesai. Lanjutkan dengan: python src/check_dataset.py")

if __name__ == "__main__":
    main()
