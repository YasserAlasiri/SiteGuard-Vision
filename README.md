# SiteGuard Vision — Construction PPE Compliance & Restricted-Zone Analytics

**Ultralytics YOLO Computer Vision Capstone Project**
Student name: MOHAMMED ABDULLAH ALHAMMADI , YASSER AHMED ALASSIRI , Faisal Saleh BinBaz , Asaad Jamal Maqbool , SAAD MOHAMMED ALHOSAN

SiteGuard Vision is an end-to-end computer vision solution designed to improve safety monitoring on construction sites. The system uses Ultralytics YOLO to detect personal protective equipment (PPE), evaluate a custom-trained model, perform instance segmentation, track workers in video, monitor restricted zones, identify missing-PPE conditions, export the trained model to ONNX, and provide a simple Streamlit deployment interface.

## Project Objectives

The project aims to:

- Detect workers and PPE items using YOLO.
- Train a YOLO model on a real construction PPE dataset.
- Evaluate model performance using standard detection metrics.
- Perform instance segmentation as an additional computer vision task.
- Track workers across video frames using ByteTrack.
- Count workers entering a defined restricted zone.
- Detect missing PPE inside the restricted zone.
- Export the trained model to ONNX for deployment.
- Provide a simple Streamlit application for image inference.

## Dataset

The project uses the **Ultralytics Construction-PPE dataset** for custom object detection training.

Dataset characteristics:

- Total images: **1,416**
- Training images: **1,132**
- Validation images: **143**
- Test images: **141**
- Number of classes: **11**

Classes:

```text
helmet
gloves
vest
boots
goggles
none
Person
no_helmet
no_goggle
no_gloves
no_boots
```

Ultralytics automatically downloads the dataset when `construction-ppe.yaml` is used during training.

Dataset documentation:
https://docs.ultralytics.com/datasets/detect/construction-ppe/

## Models

The project uses:

- `yolo26n.pt` for custom PPE detection training.
- `yolo26n-seg.pt` for instance segmentation.
- The custom trained `best.pt` model for evaluation, inference, tracking, analytics, and export.
- ByteTrack for persistent object tracking.

The trained detection model is expected at:

```text
runs/siteguard/ppe_train/weights/best.pt
```

## Project Structure

```text
SiteGuard_Vision_Capstone/
├── app.py
├── run_pipeline.py
├── requirements.txt
├── configs/
│   └── project.yaml
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

## Requirements

Recommended environment:

- Python 3.10–3.12
- Ultralytics
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Streamlit
- ONNX

A GPU is recommended for model training.

Install dependencies with:

```bash
pip install -r requirements.txt
```

## 1. Train the Custom YOLO Model

Run:

```bash
python src/train.py
```

The training pipeline fine-tunes a pretrained YOLO model using the Construction-PPE dataset.

Main training configuration:

- Image size: 640
- Batch size: 16
- Epochs: 50
- Patience: 12
- Seed: 42
- Pretrained weights enabled

Expected model output:

```text
runs/siteguard/ppe_train/weights/best.pt
```

## 2. Model Evaluation

Run:

```bash
python src/evaluate.py
```

The evaluation script uses Ultralytics `model.val()` and reports:

- Precision
- Recall
- mAP@50
- mAP@50-95

Evaluation results are saved to:

```text
artifacts/evaluation_metrics.json
```

Validation plots and confusion-matrix outputs are saved under the Ultralytics run directory.

### Final Evaluation Results

Replace the values below with the real values generated after executing the evaluation script.

| Metric | Result |
|---|---:|
| Precision | `ADD_REAL_VALUE` |
| Recall | `ADD_REAL_VALUE` |
| mAP@50 | `ADD_REAL_VALUE` |
| mAP@50-95 | `ADD_REAL_VALUE` |

### Evaluation Interpretation

Precision measures how many predicted detections are correct, while recall measures how many real objects were successfully detected. In a construction safety application, recall is especially important because missed safety violations may result in hazards not being detected.

The selected confidence threshold should balance false positives and false negatives. Lower thresholds can improve recall but may produce more false alarms, while higher thresholds can reduce false alarms but may miss important safety violations.

## 3. Training Analysis

Run:

```bash
python src/analyze_training.py
```

The script reads the real Ultralytics training history from `results.csv` and generates a training analysis file:

```text
artifacts/training_analysis.md
```

The analysis should be used to discuss:

- Training convergence
- Validation performance
- Best validation epoch
- Possible overfitting
- Possible underfitting

The final conclusion should be based on the actual generated training curves and validation metrics.

## 4. Detection Inference

Run custom object detection on an image:

```bash
python src/inference.py --source path/to/construction_image.jpg
```

The script loads the trained custom model and performs YOLO inference on the provided image.

Annotated results are saved under:

```text
runs/siteguard/inference/
```

## 5. Instance Segmentation

The project includes instance segmentation as an additional computer vision task.

Run:

```bash
python src/segmentation_demo.py --source path/to/construction_image.jpg
```

The script loads:

```text
yolo26n-seg.pt
```

and performs segmentation using `model.predict()`.

Generated segmentation results are saved under:

```text
runs/siteguard/segmentation/
```

## 6. Video Tracking and Restricted-Zone Analytics

Run the video analytics pipeline on a construction video:

```bash
python src/video_analytics.py --source path/to/construction_site.mp4 --show
```

A webcam can also be used:

```bash
python src/video_analytics.py --source 0 --show
```

The video pipeline performs the following operations:

1. Reads video frames using OpenCV.
2. Runs YOLO detection and object tracking.
3. Uses ByteTrack to maintain persistent track IDs.
4. Defines a polygon-based restricted work zone.
5. Counts unique workers entering the zone.
6. Detects missing PPE conditions inside the zone.
7. Draws tracking information and alerts on the video.
8. Saves the processed video and analytics summary.

Expected outputs:

```text
artifacts/siteguard_analytics.mp4
artifacts/video_analytics_summary.json
```

## 7. ONNX Model Export

Run:

```bash
python src/export_model.py
```

The script exports the trained YOLO model using:

```python
model.export(format="onnx")
```

The generated ONNX model can be used for deployment in compatible inference environments.

## 8. Streamlit Deployment

Run the application with:

```bash
streamlit run app.py
```

The application allows the user to upload a construction image and view predictions generated by the trained PPE detection model.

## Complete Pipeline

The main model workflow can be executed with:

```bash
python run_pipeline.py
```

This pipeline performs:

- Training
- Evaluation
- Training analysis
- ONNX export

Image segmentation and video analytics are executed separately because they require user-provided image or video inputs.

## Execution Evidence

The final repository should include evidence generated from real execution of the project.

Recommended evidence includes:

- Completed training output
- `best.pt`
- Training curves
- Precision, Recall, mAP@50, and mAP@50-95 values
- Confusion matrix
- Detection inference output image
- Segmentation output image
- Processed video analytics result
- Video analytics JSON summary
- ONNX export output
- Executed notebook cells with visible outputs

Generated metric values should come directly from the executed model and should not be manually invented.

## Example Result Files

After successful execution, the repository may contain generated outputs such as:

```text
artifacts/
├── evaluation_metrics.json
├── training_analysis.md
├── video_analytics_summary.json
└── siteguard_analytics.mp4

runs/siteguard/
├── ppe_train/
├── evaluation/
├── inference/
└── segmentation/
```

Large datasets and unnecessary generated files should not be committed to GitHub.

## GitHub Repository Practices

The repository should use a clear project structure and meaningful commits.

Example commit messages:

```text
feat: initialize SiteGuard Vision project
feat: add YOLO PPE training pipeline
feat: add model evaluation workflow
feat: add segmentation inference
feat: add worker tracking and zone analytics
feat: add ONNX export
feat: add Streamlit deployment
fix: update execution results and documentation
docs: add final project results
```

Do not commit API keys, credentials, private information, datasets, or unnecessary temporary files.

## Training Program Attribution

This project was developed as part of the **Computer Vision for Developers with Ultralytics** training program delivered by **SDAIA Academy** through **Learning Space**.

SDAIA Academy GitHub:
https://github.com/SDAIAAcademy

## Technical Documentation

Additional technical information is available in:

```text
docs/TECHNICAL_DOCUMENTATION.md
```

The document describes the system architecture, dataset and model choices, evaluation methodology, video analytics workflow, deployment approach, and reproducibility considerations.

## Limitations

SiteGuard Vision is an educational computer vision project and should not be considered a certified occupational safety system. Real-world deployment would require additional dataset validation, environmental testing, privacy controls, model calibration, and operational safety procedures.
