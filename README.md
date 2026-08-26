# SiteGuard Vision — Construction PPE Compliance & Restricted-Zone Analytics
Student name: MOHAMMED ABDULLAH ALHAMMADI , YASSER AHMED ALASSIRI , SAAD MOHAMMED ALHOSAN , Asaad Jamal Maqbool , Faisal Saleh BinBaz


**Ultralytics YOLO Capstone Computer Vision Solution**

SiteGuard Vision is an end-to-end computer-vision application for construction safety. It fine-tunes a YOLO detector on a real PPE dataset, evaluates the model, runs custom inference, performs **instance segmentation as a second vision task**, tracks workers and equipment in video, counts people entering a restricted work zone, flags missing PPE inside that zone, exports the trained detector to ONNX, and provides a small Streamlit deployment app.

## Why this project
Manual PPE monitoring is difficult across busy sites. SiteGuard Vision turns a camera stream into actionable safety analytics: worker/PPE detection, persistent tracking, restricted-zone entry counts, and visible missing-PPE alerts. The system is a capstone demonstration, not a certified safety system.

## Scored requirements covered
| Rubric deliverable | Implementation |
|---|---|
| Core Vision Tasks & Inference (25) | `src/inference.py` + `src/segmentation_demo.py`; real YOLO predict calls; segmentation uses `yolo26n-seg.pt` |
| Real-World Solution & Video Analytics (25) | `src/video_analytics.py`; `model.track()`, ByteTrack, OpenCV video pipeline, polygon zone analytics, entry counting, PPE alerts |
| Model Evaluation (25) | `src/evaluate.py`; real `model.val()`, mAP50, mAP50-95, precision, recall, plots/confusion matrix, threshold interpretation |
| Custom Data & Training (15) | `src/train.py`; real `model.train()` on Construction-PPE, 1,416 images, 11 classes; not COCO8 |
| Deployment & Export (5) | `src/export_model.py` exports ONNX; `app.py` provides Streamlit inference |
| Documentation & Evidence (5) | This README + `docs/`; exact evidence checklist and GitHub requirements |

**Section 2.3 “Encouraged: supporting the Saudi tech community” is intentionally not included, as requested.** All mandatory GitHub requirements in Sections 2.1 and 2.2 remain covered.

## Dataset
The custom-training dataset is **Ultralytics Construction-PPE**, a real construction-site detection dataset containing **1,416 images**, split into **1,132 train / 143 validation / 141 test**, with **11 classes**: `helmet`, `gloves`, `vest`, `boots`, `goggles`, `none`, `Person`, `no_helmet`, `no_goggle`, `no_gloves`, `no_boots`.

Ultralytics downloads the dataset automatically when `data="construction-ppe.yaml"` is used. Dataset documentation: `https://docs.ultralytics.com/datasets/detect/construction-ppe/`.

## Models
- Fine-tuning / custom PPE detection: `yolo26n.pt`
- Second task — instance segmentation: `yolo26n-seg.pt`
- Final custom weights after training: `runs/siteguard/ppe_train/weights/best.pt`
- Tracking: custom `best.pt` + `bytetrack.yaml`

> If your installed Ultralytics release uses a different current nano checkpoint name, replace the model strings consistently. The APIs used are the standard Ultralytics Python `train`, `val`, `predict`, `track`, and `export` workflows.

## Project structure
```text
SiteGuard_Vision_Capstone/
├── app.py
├── run_pipeline.py
├── requirements.txt
├── configs/project.yaml
├── src/
│   ├── train.py
│   ├── evaluate.py
│   ├── analyze_training.py
│   ├── inference.py
│   ├── segmentation_demo.py
│   ├── video_analytics.py
│   └── export_model.py
├── docs/
│   ├── TECHNICAL_DOCUMENTATION.md
│   ├── RUBRIC_CHECKLIST.md
│   └── EXECUTION_EVIDENCE.md
├── artifacts/
└── tests/
```

## Prerequisites
- Python 3.10–3.12 recommended
- Internet access on first run to install dependencies, download pretrained model weights, and download Construction-PPE
- GPU recommended for training; CPU works but is much slower
- A real construction-site video or webcam stream for Deliverable 2 evidence

## Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## 1) Train on real custom data
```bash
python src/train.py
```
Default training: 50 epochs, image size 640, batch 16, patience 12, pretrained nano weights, deterministic seed 42. Ultralytics will auto-download `construction-ppe.yaml` data if needed.

Expected key artifact:
```text
runs/siteguard/ppe_train/weights/best.pt
```

## 2) Evaluate the trained model
```bash
python src/evaluate.py
```
This **actually calls `model.val()`** and saves concrete metrics to:
```text
artifacts/evaluation_metrics.json
```
It reports mAP50, mAP50-95, mean precision, mean recall, and preserves the chosen thresholds (`conf=0.35`, `IoU=0.50`). Ultralytics validation plots should be under `runs/siteguard/evaluation/`.

### Evaluation interpretation
For construction safety, false negatives are especially important because a missed missing-PPE condition may fail to warn about risk. Using confidence 0.35 is a recall-favoring starting point. A threshold that is too low can create false-positive alerts; a threshold that is too high can miss hazards. Use the real PR/F1 curves and confusion matrix to tune this trade-off. IoU 0.50 is a practical overlap threshold for deciding whether predicted and labeled boxes align sufficiently.

## 3) Analyze what happened during training
```bash
python src/analyze_training.py
```
This reads the **real** `results.csv` and writes:
```text
artifacts/training_analysis.md
```
The analysis identifies the best validation epoch and checks for a late-stage validation-mAP drop, so the required over/underfitting discussion is based on evidence rather than guessed values.

## 4) Core custom detection inference
```bash
python src/inference.py --source path/to/construction_image.jpg
```
Annotated output and prediction labels are saved under `runs/siteguard/inference/`.

## 5) Required task beyond detection — instance segmentation
```bash
python src/segmentation_demo.py --source path/to/construction_image.jpg
```
This loads task-specific segmentation weights (`yolo26n-seg.pt`) and executes real instance segmentation with `model.predict()`. Annotated masks are saved under `runs/siteguard/segmentation/`.

## 6) Real-world video tracking & zone analytics
Use a real construction-site clip:
```bash
python src/video_analytics.py --source path/to/construction_site.mp4 --show
```
Or a webcam:
```bash
python src/video_analytics.py --source 0 --show
```
The pipeline does real work on every frame:
1. OpenCV captures the stream.
2. `model.track(..., persist=True, tracker="bytetrack.yaml")` produces persistent object tracks.
3. A polygon defines a restricted work zone.
4. `Person` track IDs entering that polygon are counted once.
5. `no_helmet`, `no_goggle`, `no_gloves`, and `no_boots` detections inside the polygon are highlighted as alerts.
6. The processed video and JSON analytics summary are written to `artifacts/`.

Outputs:
```text
artifacts/siteguard_analytics.mp4
artifacts/video_analytics_summary.json
```

## 7) Export for deployment
```bash
python src/export_model.py
```
This calls:
```python
model.export(format="onnx")
```
and creates an optimized ONNX model next to the trained weights.

## 8) Run the Streamlit app
```bash
streamlit run app.py
```
Upload a construction image to display detections from the custom model.

## Optional one-command scored pipeline
Training + evaluation + training analysis + ONNX export:
```bash
python run_pipeline.py
```
Segmentation and video analytics are intentionally separate because they require real user-provided image/video evidence.

## Evidence required before submission
The rubric says unexecuted notebooks/code prove nothing. Follow `docs/EXECUTION_EVIDENCE.md` and retain real terminal output, generated metrics, plots, annotated images, video analytics output, and export evidence. **Do not fabricate scores.**

## GitHub submission requirements
This project is structured for a proper GitHub repository. Before submission:
- Create/activate your own GitHub account.
- Publish the project to GitHub; a project not published as required is incomplete.
- Use meaningful incremental commits, not one bulk upload.
- Keep a sensible repository structure.
- Do not commit secrets, API keys, datasets, weights, `runs/`, or large generated media; `.gitignore` is already configured.

Suggested commit sequence:
```text
feat: initialize SiteGuard Vision project structure
feat: add Construction-PPE YOLO training pipeline
feat: add validation metrics and training analysis
feat: add segmentation inference task
feat: add tracking and restricted-zone analytics
feat: add ONNX export and Streamlit deployment
 docs: complete capstone README and evidence checklist
```

## Training program attribution
This capstone was created for the **Computer Vision for Developers with Ultralytics** training program delivered by **SDAIA Academy** via **Learning Space** as a 5-day capstone program.

SDAIA Academy on GitHub: **https://github.com/SDAIAAcademy**

## Technical notes
See `docs/TECHNICAL_DOCUMENTATION.md` for pipeline design, model/data choices, evaluation reasoning, video architecture, deployment, and reproducibility details. See `docs/RUBRIC_CHECKLIST.md` for a one-to-one rubric mapping.

## Important limitation
This repository is an educational capstone. It should not be treated as a certified occupational-safety monitoring system without site-specific validation, privacy review, calibration, and operational safeguards.
