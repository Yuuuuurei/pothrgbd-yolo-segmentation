"""
evaluate_segmentation.py
------------------------
Evaluasi model YOLOv8-seg pada test set PothRGBD.

Cara pakai:
    python src/evaluate_segmentation.py
    python src/evaluate_segmentation.py --weights runs/segment/pothrgbd_seg/weights/best.pt
    python src/evaluate_segmentation.py --split val
"""

import argparse
import json
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="Evaluasi YOLOv8 segmentation — PothRGBD")
    p.add_argument("--weights", default="runs/segment/pothrgbd_seg/weights/best.pt")
    p.add_argument("--data",    default="data/pothrgbd/data.yaml")
    p.add_argument("--split",   default="test", choices=["train", "val", "test"])
    p.add_argument("--imgsz",   type=int, default=640)
    p.add_argument("--batch",   type=int, default=8)
    p.add_argument("--conf",    type=float, default=0.001)  # rendah untuk mAP
    p.add_argument("--iou",     type=float, default=0.6)
    p.add_argument("--device",  default="")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  PothRGBD — Segmentation Evaluation")
    print("=" * 60)
    print(f"  Weights : {args.weights}")
    print(f"  Split   : {args.split}")
    print()

    if not Path(args.weights).exists():
        print(f"[ERROR] Weights tidak ditemukan: {args.weights}")
        return

    model = YOLO(args.weights)

    metrics = model.val(
        data   = args.data,
        split  = args.split,
        imgsz  = args.imgsz,
        batch  = args.batch,
        conf   = args.conf,
        iou    = args.iou,
        device = args.device if args.device else None,
        verbose= True,
    )

    # ── Ekstrak dan tampilkan metrik utama ──────────────────────────────────
    results_dict = {}

    # Box metrics
    if hasattr(metrics, "box"):
        box = metrics.box
        results_dict["box"] = {
            "precision" : float(box.mp),
            "recall"    : float(box.mr),
            "mAP50"     : float(box.map50),
            "mAP50_95"  : float(box.map),
        }
        print("\n── Box Metrics ──────────────────────────────────────")
        print(f"  Precision  : {box.mp:.4f}")
        print(f"  Recall     : {box.mr:.4f}")
        print(f"  mAP@50     : {box.map50:.4f}")
        print(f"  mAP@50-95  : {box.map:.4f}")

    # Mask metrics
    if hasattr(metrics, "seg"):
        seg = metrics.seg
        results_dict["mask"] = {
            "precision" : float(seg.mp),
            "recall"    : float(seg.mr),
            "mAP50"     : float(seg.map50),
            "mAP50_95"  : float(seg.map),
        }
        print("\n── Mask Metrics ─────────────────────────────────────")
        print(f"  Precision  : {seg.mp:.4f}")
        print(f"  Recall     : {seg.mr:.4f}")
        print(f"  mAP@50     : {seg.map50:.4f}")
        print(f"  mAP@50-95  : {seg.map:.4f}")

    # ── Simpan ke file ───────────────────────────────────────────────────────
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    summary_path = out_dir / "metrics_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"\n[OK] Metrik disimpan ke {summary_path}")

    txt_path = out_dir / "metrics_summary.txt"
    with open(txt_path, "w") as f:
        f.write(f"Split   : {args.split}\n")
        f.write(f"Weights : {args.weights}\n\n")
        for group, vals in results_dict.items():
            f.write(f"[{group.upper()}]\n")
            for k, v in vals.items():
                f.write(f"  {k:<12}: {v:.4f}\n")
            f.write("\n")
    print(f"[OK] Ringkasan teks disimpan ke {txt_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
