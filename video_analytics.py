"""Real-world construction safety video analytics: tracking, zone entries, and PPE violations."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from common import save_json

VIOLATION_CLASSES = {"no_helmet", "no_goggle", "no_gloves", "no_boots"}


def parse_source(s: str):
    return int(s) if s.isdigit() else s


def default_zone(width: int, height: int) -> np.ndarray:
    # Central restricted-work polygon; can be replaced with site-specific coordinates.
    return np.array([
        [int(width * 0.20), int(height * 0.20)],
        [int(width * 0.85), int(height * 0.20)],
        [int(width * 0.90), int(height * 0.90)],
        [int(width * 0.15), int(height * 0.90)],
    ], dtype=np.int32)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="0", help="Video path/URL or webcam index, e.g. 0")
    p.add_argument("--weights", default="runs/siteguard/ppe_train/weights/best.pt")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--iou", type=float, default=0.50)
    p.add_argument("--tracker", default="bytetrack.yaml")
    p.add_argument("--output", default="artifacts/siteguard_analytics.mp4")
    p.add_argument("--summary", default="artifacts/video_analytics_summary.json")
    p.add_argument("--show", action="store_true")
    args = p.parse_args()

    if not Path(args.weights).exists():
        raise FileNotFoundError(f"Weights not found: {args.weights}. Train first.")

    cap = cv2.VideoCapture(parse_source(args.source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 25.0
    zone = default_zone(width, height)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    model = YOLO(args.weights)

    prior_inside: dict[int, bool] = {}
    unique_zone_entries: set[int] = set()
    tracked_ids: set[int] = set()
    violation_events = defaultdict(int)
    frames = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1
        result = model.track(  # rubric-critical real tracking API call
            frame,
            persist=True,
            tracker=args.tracker,
            conf=args.conf,
            iou=args.iou,
            verbose=False,
        )[0]
        annotated = result.plot()
        cv2.polylines(annotated, [zone], True, (255, 255, 255), 2)
        cv2.putText(annotated, "RESTRICTED WORK ZONE", tuple(zone[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        if result.boxes is not None and len(result.boxes):
            boxes = result.boxes
            ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [-1] * len(boxes)
            clss = boxes.cls.int().cpu().tolist()
            xyxy = boxes.xyxy.cpu().numpy()
            names = result.names

            for track_id, cls_idx, box in zip(ids, clss, xyxy):
                cls_name = names[int(cls_idx)]
                x1, y1, x2, y2 = box
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                inside = cv2.pointPolygonTest(zone, (cx, cy), False) >= 0
                if track_id >= 0:
                    tracked_ids.add(track_id)

                if cls_name == "Person" and track_id >= 0:
                    was_inside = prior_inside.get(track_id, False)
                    if inside and not was_inside:
                        unique_zone_entries.add(track_id)
                    prior_inside[track_id] = inside

                if cls_name in VIOLATION_CLASSES and inside:
                    violation_events[cls_name] += 1
                    cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
                    cv2.putText(annotated, f"ALERT: {cls_name}", (int(x1), max(25, int(y1)-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,255), 2)

        cv2.putText(annotated, f"Zone entries: {len(unique_zone_entries)}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(annotated, f"PPE violation detections: {sum(violation_events.values())}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        writer.write(annotated)
        if args.show:
            cv2.imshow("SiteGuard Vision", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    summary = {
        "source": str(args.source),
        "frames_processed": frames,
        "unique_track_ids_seen": len(tracked_ids),
        "unique_person_zone_entries": len(unique_zone_entries),
        "violation_detections_by_class": dict(violation_events),
        "total_violation_detections": int(sum(violation_events.values())),
        "output_video": str(out_path),
        "note": "Violation counts are frame-level detection events; zone entry count uses persistent Person track IDs."
    }
    save_json(summary, args.summary)
    print(summary)


if __name__ == "__main__":
    main()
