"""Run fine-tuned PPE detection inference and save actual predictions."""
from __future__ import annotations

import argparse
from pathlib import Path
from ultralytics import YOLO


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--weights", default="runs/siteguard/ppe_train/weights/best.pt")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--iou", type=float, default=0.50)
    args = p.parse_args()

    if not Path(args.weights).exists():
        raise FileNotFoundError(f"Weights not found: {args.weights}")
    model = YOLO(args.weights)
    results = model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        save=True,
        save_txt=True,
        save_conf=True,
        project="runs/siteguard",
        name="inference",
        exist_ok=True,
    )
    detections = sum(0 if r.boxes is None else len(r.boxes) for r in results)
    print(f"Inference complete. Detections: {detections}")
    print("Annotated output: runs/siteguard/inference/")


if __name__ == "__main__":
    main()
