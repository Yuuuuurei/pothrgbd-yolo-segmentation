"""
predict_segmentation.py
-----------------------
Jalankan inference YOLOv8-seg pada gambar test dan simpan hasil prediksi.

Cara pakai:
    python src/predict_segmentation.py
    python src/predict_segmentation.py --source path/ke/gambar.jpg
    python src/predict_segmentation.py --source data/pothrgbd/test/images --conf 0.40
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO


IMG_EXTS = {".jpg", ".jpeg", ".png"}


def parse_args():
    p = argparse.ArgumentParser(description="Prediksi segmentasi pothole")
    p.add_argument("--weights", default="runs/segment/pothrgbd_seg/weights/best.pt")
    p.add_argument("--source",  default="data/pothrgbd/test/images")
    p.add_argument("--conf",    type=float, default=0.25)
    p.add_argument("--iou",     type=float, default=0.45)
    p.add_argument("--imgsz",   type=int,   default=640)
    p.add_argument("--device",  default="")
    p.add_argument("--save-samples", type=int, default=10,
                   help="Jumlah sampel gambar yang disimpan ke outputs/sample_masks/")
    return p.parse_args()


def overlay_mask_on_image(image: np.ndarray, masks: np.ndarray,
                           color=(0, 255, 100), alpha=0.45) -> np.ndarray:
    """
    Overlay mask biner ke atas gambar dengan warna dan transparansi.
    masks: (N, H, W) boolean/float array
    """
    overlay = image.copy()
    for mask in masks:
        binary = (mask > 0.5).astype(np.uint8)
        colored = np.zeros_like(image)
        colored[:] = color
        mask_3ch = np.stack([binary] * 3, axis=-1)
        overlay = np.where(mask_3ch, cv2.addWeighted(overlay, 1 - alpha,
                                                      colored, alpha, 0), overlay)
    return overlay


def main():
    args = parse_args()

    print("=" * 60)
    print("  PothRGBD — Segmentation Inference")
    print("=" * 60)

    if not Path(args.weights).exists():
        print(f"[ERROR] Weights tidak ditemukan: {args.weights}")
        return

    model = YOLO(args.weights)

    # ── Jalankan prediksi ───────────────────────────────────────────────────
    results = model.predict(
        source  = args.source,
        conf    = args.conf,
        iou     = args.iou,
        imgsz   = args.imgsz,
        device  = args.device if args.device else None,
        save    = True,
        name    = "pothrgbd_predictions",
    )

    # ── Simpan sampel dengan overlay custom ─────────────────────────────────
    sample_dir = Path("outputs/sample_masks")
    sample_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for r in results:
        if saved >= args.save_samples:
            break

        img_path = Path(r.path)
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        if r.masks is not None and len(r.masks) > 0:
            masks_np = r.masks.data.cpu().numpy()   # (N, H, W)
            # resize masks ke ukuran gambar asli
            h, w = img.shape[:2]
            masks_resized = np.array([
                cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
                for m in masks_np
            ])
            vis = overlay_mask_on_image(img, masks_resized)

            # Anotasi confidence di setiap instance
            for i, box in enumerate(r.boxes):
                conf_val = float(box.conf)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 255), 2)
                cv2.putText(vis, f"pothole {conf_val:.2f}",
                            (x1, max(y1 - 8, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)

            out_path = sample_dir / f"sample_{img_path.stem}.jpg"
            cv2.imwrite(str(out_path), vis)
            saved += 1

    print(f"\n[OK] {saved} sampel mask tersimpan di: {sample_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
