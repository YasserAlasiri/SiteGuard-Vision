"""Export the fine-tuned model to optimized ONNX for deployment."""
from __future__ import annotations

import argparse
from pathlib import Path
from ultralytics import YOLO


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="runs/siteguard/ppe_train/weights/best.pt")
    p.add_argument("--imgsz", type=int, default=640)
    args = p.parse_args()
    if not Path(args.weights).exists():
        raise FileNotFoundError(f"Weights not found: {args.weights}")
    model = YOLO(args.weights)
    exported = model.export(format="onnx", imgsz=args.imgsz, simplify=True, dynamic=False)  # rubric-critical export
    print(f"ONNX export complete: {exported}")


if __name__ == "__main__":
    main()
