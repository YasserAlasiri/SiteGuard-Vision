"""Required task beyond plain detection: instance segmentation."""
from __future__ import annotations

import argparse
from pathlib import Path
from ultralytics import YOLO


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, help="Image, video, webcam index, or URL")
    p.add_argument("--model", default="yolo26n-seg.pt")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--output-dir", default="runs/siteguard/segmentation")
    args = p.parse_args()

    model = YOLO(args.model)  # task-specific segmentation weights
    results = model.predict(  # real inference API call
        source=args.source,
        conf=args.conf,
        save=True,
        project=str(Path(args.output_dir).parent),
        name=Path(args.output_dir).name,
        exist_ok=True,
    )
    n_masks = sum(0 if r.masks is None else len(r.masks.data) for r in results)
    print(f"Segmentation complete. Total instance masks: {n_masks}")
    print(f"Saved annotated output to {args.output_dir}")


if __name__ == "__main__":
    main()
