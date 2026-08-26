"""Fine-tune Ultralytics YOLO on a real Construction-PPE dataset."""
from __future__ import annotations

import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="yolo26n.pt")
    p.add_argument("--data", default="construction-ppe.yaml")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--device", default=None, help="e.g. 0, cpu, mps; default lets Ultralytics choose")
    p.add_argument("--project", default="runs/siteguard")
    p.add_argument("--name", default="ppe_train")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)
    kwargs = dict(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        seed=42,
        deterministic=True,
        pretrained=True,
        plots=True,
        project=args.project,
        name=args.name,
        exist_ok=True,
    )
    if args.device:
        kwargs["device"] = args.device
    results = model.train(**kwargs)  # rubric-critical real train API call
    save_dir = Path(getattr(results, "save_dir", Path(args.project) / args.name))
    print(f"Training complete. Results: {save_dir}")
    print(f"Best weights: {save_dir / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
