"""
train_segmentation.py
---------------------
Training YOLOv8 segmentation untuk deteksi pothole pada PothRGBD dataset.

Cara pakai:
    python src/train_segmentation.py
    python src/train_segmentation.py --model yolov8s-seg --epochs 100
    python src/train_segmentation.py --resume runs/segment/pothrgbd_yolov8n_seg/weights/last.pt
"""

import argparse
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ── Default Config ─────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "model"    : "yolov8n-seg.pt",
    "data"     : PROJECT_ROOT / "data" / "pothrgbd" / "data.yaml",
    "epochs"   : 60,
    "imgsz"    : 640,
    "batch"    : 8,
    "patience" : 10,
    "workers"  : 2,
    "device"   : "",          # "" → auto detect (GPU jika ada, CPU jika tidak)
    "project"  : PROJECT_ROOT / "runs" / "segment",
    "name"     : "pothrgbd_seg",
    "exist_ok" : False,
    "amp"      : True,        # Automatic Mixed Precision (hemat VRAM)
    "cache"    : False,       # True = cache ke RAM (lebih cepat, butuh RAM besar)
    "augment"  : True,
}
# ───────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="Train YOLOv8 segmentation — PothRGBD")
    p.add_argument("--model",    default=DEFAULT_CONFIG["model"],
                   help="Model checkpoint, e.g. yolov8n-seg.pt / yolov8s-seg.pt")
    p.add_argument("--data",     default=DEFAULT_CONFIG["data"])
    p.add_argument("--epochs",   type=int, default=DEFAULT_CONFIG["epochs"])
    p.add_argument("--imgsz",    type=int, default=DEFAULT_CONFIG["imgsz"])
    p.add_argument("--batch",    type=int, default=DEFAULT_CONFIG["batch"])
    p.add_argument("--patience", type=int, default=DEFAULT_CONFIG["patience"])
    p.add_argument("--workers",  type=int, default=DEFAULT_CONFIG["workers"])
    p.add_argument("--device",   default=DEFAULT_CONFIG["device"],
                   help="'0' untuk GPU 0, 'cpu', atau '' untuk auto")
    p.add_argument("--name",     default=DEFAULT_CONFIG["name"])
    p.add_argument("--resume",   default=None,
                   help="Path ke last.pt untuk lanjut training")
    p.add_argument("--no-amp",   action="store_true", help="Matikan AMP")
    p.add_argument("--cache",    action="store_true", help="Cache dataset ke RAM")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  PothRGBD — YOLOv8 Segmentation Training")
    print("=" * 60)

    if args.resume:
        print(f"[INFO] Resume dari checkpoint: {args.resume}")
        model = YOLO(args.resume)
    else:
        print(f"[INFO] Model: {args.model}")
        model = YOLO(args.model)

    print(f"[INFO] Data    : {args.data}")
    print(f"[INFO] Epochs  : {args.epochs}")
    print(f"[INFO] Image   : {args.imgsz}")
    print(f"[INFO] Batch   : {args.batch}")
    print(f"[INFO] Device  : {args.device or 'auto'}")
    print()

    results = model.train(
        data     = str(Path(args.data).resolve() if Path(args.data).is_absolute() else PROJECT_ROOT / args.data),
        epochs   = args.epochs,
        imgsz    = args.imgsz,
        batch    = args.batch,
        patience = args.patience,
        workers  = args.workers,
        device   = args.device if args.device else None,
        project  = str(DEFAULT_CONFIG["project"]),
        name     = args.name,
        exist_ok = DEFAULT_CONFIG["exist_ok"],
        amp      = not args.no_amp,
        cache    = args.cache,
        augment  = DEFAULT_CONFIG["augment"],
        resume   = bool(args.resume),
    )

    best_weights = Path(DEFAULT_CONFIG["project"]) / args.name / "weights" / "best.pt"
    print("\n" + "=" * 60)
    print(f"  Training selesai.")
    print(f"  Best weights: {best_weights}")
    print("=" * 60)
    return results


if __name__ == "__main__":
    main()
