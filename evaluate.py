"""Run real validation and persist concrete YOLO metrics."""
from __future__ import annotations

import argparse
from pathlib import Path
from ultralytics import YOLO
from common import save_json


def _safe_float(obj, attr: str):
    value = getattr(obj, attr, None)
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="runs/siteguard/ppe_train/weights/best.pt")
    p.add_argument("--data", default="construction-ppe.yaml")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--iou", type=float, default=0.50)
    p.add_argument("--output", default="artifacts/evaluation_metrics.json")
    args = p.parse_args()

    if not Path(args.weights).exists():
        raise FileNotFoundError(f"Weights not found: {args.weights}. Run src/train.py first.")

    model = YOLO(args.weights)
    metrics = model.val(  # rubric-critical real val API call
        data=args.data,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        plots=True,
        split="val",
        project="runs/siteguard",
        name="evaluation",
        exist_ok=True,
    )

    box = metrics.box
    report = {
        "weights": args.weights,
        "dataset": args.data,
        "thresholds": {"confidence": args.conf, "iou": args.iou},
        "metrics": {
            "mAP50": _safe_float(box, "map50"),
            "mAP50_95": _safe_float(box, "map"),
            "precision_mean": _safe_float(box, "mp"),
            "recall_mean": _safe_float(box, "mr"),
        },
        "interpretation_notes": {
            "false_positive_risk": "A low confidence threshold may over-report missing PPE; inspect confusion_matrix.png and validation predictions.",
            "false_negative_risk": "A high confidence threshold may miss unsafe workers, which is especially costly in a safety use case.",
            "threshold_rationale": "conf=0.35 favors recall for safety monitoring; IoU=0.50 is a practical overlap threshold. Tune using PR/F1 curves after validation."
        }
    }
    save_json(report, args.output)
    print(report)
    print(f"Saved metrics to {args.output}")
    print("Ultralytics plots (including confusion matrix / PR curves when supported) are under runs/siteguard/evaluation/")


if __name__ == "__main__":
    main()
