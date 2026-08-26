# Capstone Rubric Coverage Checklist

> Section **2.3 Encouraged: supporting the Saudi tech community is intentionally excluded**, per project instruction. It is not part of the scored 100-point rubric.

## Deliverable 1 — Core Vision Tasks & Inference (25)
- [x] Ultralytics YOLO is loaded and called through the Python API.
- [x] `src/inference.py` performs real custom detection inference and saves annotated output and labels.
- [x] A task beyond plain detection is implemented: instance segmentation in `src/segmentation_demo.py` using task-specific `yolo26n-seg.pt` weights.
- [ ] After execution, commit/capture a representative inference screenshot/log as evidence.

## Deliverable 2 — Real-World Solution & Video Analytics (25)
- [x] Real construction-safety use case rather than isolated stock-image inference.
- [x] `model.track()` is actually called with persistent tracking IDs and ByteTrack.
- [x] OpenCV capture-process-write/display pipeline over video or webcam.
- [x] Restricted-region analytics and unique Person zone-entry counting.
- [x] Missing-PPE alert logic and JSON summary.
- [ ] Run on a real construction-site video/webcam clip and keep the output MP4 + terminal log/screenshot.

## Deliverable 3 — Model Evaluation (25)
- [x] `model.val()` is actually called on the real validation split.
- [x] mAP50, mAP50-95, mean precision, mean recall are persisted to JSON.
- [x] Evaluation plots requested, including confusion matrix / PR curves when available.
- [x] False-positive vs false-negative impact is documented for a safety use case.
- [x] Confidence=0.35 and IoU=0.50 rationale is documented and can be tuned from curves.
- [ ] Generated real metrics and plots must be retained as execution evidence.

## Deliverable 4 — Custom Data & Training (15)
- [x] `model.train()` is actually called.
- [x] Real non-COCO8 dataset: Construction-PPE (1,416 images / 11 classes).
- [x] Dataset source and class/split details documented.
- [x] Real training knobs documented: epochs, image size, batch, patience, seed, pretrained weights.
- [x] `src/analyze_training.py` generates an evidence-based over/underfitting discussion from the actual results.csv.
- [ ] After training, keep `results.csv`, curves, best.pt, and generated training analysis as evidence.

## Deliverable 5 — Deployment & Export (5)
- [x] `model.export(format="onnx")` is implemented.
- [x] Streamlit app (`app.py`) serves the custom trained model for image inference.
- [ ] Execute export and verify generated ONNX file before submission.

## Deliverable 6 — Documentation & Evidence of Execution (5)
- [x] Professional README with idea, scope, prerequisites, setup, dataset, model weights, commands, expected outputs.
- [x] Technical documentation exists under `docs/`.
- [x] Training-program attribution is in README.
- [x] SDAIA Academy GitHub link is included in README.
- [x] Proper `.gitignore` excludes secrets, datasets, runs, weights, generated files.
- [ ] Final submission must contain captured outputs/logs from an actual run. Do not claim unexecuted code as evidence.

## GitHub mandatory requirements (Sections 2.1 and 2.2)
- [x] Repository structure is clear and incremental-commit plan is documented.
- [x] Comprehensive project description is visible from README.
- [x] Professional README provides exact execution steps and expected output.
- [x] Technical documentation covers models, data, training, evaluation, modules, configuration.
- [x] Training-program attribution present.
- [x] SDAIA Academy GitHub reference present.
- [ ] Trainee must create/activate their own GitHub account and publish this repository there.
- [ ] Use meaningful incremental commits rather than one bulk upload.
