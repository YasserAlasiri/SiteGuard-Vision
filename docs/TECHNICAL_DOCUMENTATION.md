# SiteGuard Vision — Technical Documentation

## 1. Problem and scope
SiteGuard Vision is an end-to-end construction-safety computer-vision system. It fine-tunes an Ultralytics YOLO detector on a real PPE dataset, validates the trained model, runs image inference, performs instance segmentation as a second vision task, tracks people/PPE in video, measures entries into a restricted work zone, flags missing-PPE detections inside that zone, and exports the trained detector to ONNX.

## 2. Models and tasks
- **Custom object detection:** `yolo26n.pt` fine-tuned on `construction-ppe.yaml`.
- **Instance segmentation (task beyond detection):** `yolo26n-seg.pt` through `model.predict()`.
- **Video tracking:** custom `best.pt` through `model.track(..., persist=True, tracker="bytetrack.yaml")`.
- **Deployment:** custom `best.pt` through `model.export(format="onnx")` and a Streamlit app.

## 3. Dataset
Ultralytics Construction-PPE is a real construction-site detection dataset with 1,416 images and 11 classes: helmet, gloves, vest, boots, goggles, none, Person, no_helmet, no_goggle, no_gloves, no_boots. The published split is 1,132 training, 143 validation, and 141 test images. Ultralytics can download it automatically when `construction-ppe.yaml` is used.

Source: Ultralytics Construction-PPE documentation. Original dataset attribution should be retained when redistributing data. The repository deliberately excludes datasets from Git through `.gitignore`.

## 4. Training design
Default run: 50 epochs, 640px image size, batch 16, pretrained nano detector, patience 12, fixed seed 42. This is intentionally a genuine fine-tuning run on Construction-PPE rather than COCO8. The training call enables plots and saves `results.csv`, curves, labels, validation predictions, `last.pt`, and `best.pt` under `runs/siteguard/ppe_train/`.

`src/analyze_training.py` reads the real `results.csv` after training and writes `artifacts/training_analysis.md` so the over/underfitting discussion is based on what actually happened rather than fabricated metrics.

## 5. Evaluation
`src/evaluate.py` invokes `model.val()` on the validation split, stores mAP50, mAP50-95, mean precision, mean recall, confidence threshold, and IoU threshold in `artifacts/evaluation_metrics.json`, and asks Ultralytics to create evaluation plots. For this safety-oriented use case, `conf=0.35` is a recall-favoring starting point. A false negative can fail to flag an unsafe worker; too-low confidence can create false alarms. Use the PR/F1 curves and confusion matrix to tune the final threshold.

## 6. Real-world video pipeline
`src/video_analytics.py` uses OpenCV capture -> per-frame YOLO tracking -> zone analytics -> annotation -> encoded output video. A central polygon represents a restricted work zone. Persistent Person track IDs are used to count unique entries. Missing-PPE classes inside the zone generate visible alerts and are accumulated in a JSON summary.

Pipeline: `Video/Webcam -> OpenCV -> YOLO track -> ByteTrack IDs -> polygon test -> violation logic -> annotated MP4 + JSON`.

## 7. Deployment
`src/export_model.py` exports `best.pt` to ONNX with `model.export(format="onnx")`. `app.py` is a small Streamlit interface that accepts an uploaded image and displays custom PPE predictions.

## 8. Reproducibility and configuration
- Python dependencies are pinned by minimum versions in `requirements.txt`.
- Training seed is 42 and deterministic training is requested.
- Model weights, datasets, generated runs, API keys/secrets, and large videos are excluded by `.gitignore`.
- Project defaults are summarized in `configs/project.yaml`.
