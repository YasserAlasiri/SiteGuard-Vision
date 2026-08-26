# SiteGuard Vision  
## Construction PPE Compliance & Restricted-Zone Analytics

SiteGuard Vision is a computer vision solution built with **Ultralytics YOLO** to support construction-site safety monitoring.

The system detects workers and personal protective equipment (PPE), analyzes video streams, tracks detected objects, and supports safety-oriented monitoring such as PPE compliance and restricted-zone awareness.

---

## Project Overview

Construction environments contain multiple safety risks, especially when workers are missing required protective equipment.

SiteGuard Vision applies computer vision to automatically identify:

- People
- Helmets
- Gloves
- Safety vests
- Boots
- Goggles
- Missing PPE classes available in the dataset

The project combines object detection, segmentation, video analytics, model evaluation, custom training, and model export in one complete workflow.

---

## Main Features

### 1. PPE Object Detection
The trained YOLO model detects workers and PPE items from images and video.

Supported dataset classes include:

- `helmet`
- `gloves`
- `vest`
- `boots`
- `goggles`
- `none`
- `Person`
- `no_helmet`
- `no_goggle`
- `no_gloves`
- `no_boots`

### 2. Instance Segmentation
A YOLO segmentation model is used as an additional computer vision task beyond standard object detection.

This demonstrates the ability to identify object regions using segmentation masks.

### 3. Video Analytics
The project processes video using OpenCV and YOLO.

The video pipeline supports:

- Object detection
- Bounding-box visualization
- Object tracking
- Persistent tracking IDs
- PPE visualization
- Worker monitoring

### 4. Model Evaluation
The trained model is evaluated using Ultralytics `model.val()`.

Evaluation includes:

- Precision
- Recall
- mAP@50
- mAP@50-95
- Confusion matrix
- Validation plots

### 5. Custom Model Training
The model was fine-tuned on the Ultralytics **Construction-PPE** dataset.

### 6. Model Export
The trained YOLO model is exported to **ONNX** for deployment and interoperability.

---

## Dataset

The project uses the Ultralytics **Construction-PPE** dataset.

Dataset summary:

- 1,416 images
- 11 classes
- Training and validation splits
- Construction-worker and PPE annotations

Dataset configuration:

```text
construction-ppe.yaml
```

---

## Model

The project uses a pretrained Ultralytics YOLO model and fine-tunes it on the Construction-PPE dataset.

Training configuration:

```text
Model: yolo26n.pt
Image size: 640
Batch size: 16
Epochs: 3
Optimizer: AdamW (automatically selected by Ultralytics)
Pretrained weights: Yes
```

---

## Training Results

Training completed successfully for **3 epochs**.

The best model was saved as:

```text
runs/siteguard/ppe_train/weights/best.pt
```

Training analysis:

| Item | Result |
|---|---:|
| Epochs completed | 3 |
| Best validation mAP@50-95 epoch | 3 |
| Best validation mAP@50-95 | 0.12953 |
| Final validation mAP@50-95 | 0.12953 |
| Final training box loss | 1.74261 |
| Final validation box loss | 1.73273 |

The training and validation box losses were close to each other, which does not show clear evidence of overfitting during this short training run.

Because the model was trained for only three epochs, the results are more consistent with **under-training** than overfitting. Additional epochs and further hyperparameter tuning may improve recall and overall mAP.

---

## Model Evaluation Results

The trained model was evaluated on the Construction-PPE validation dataset using Ultralytics `model.val()`.

| Metric | Result |
|---|---:|
| Precision | **0.3470** |
| Recall | **0.1354** |
| mAP@50 | **0.1138** |
| mAP@50-95 | **0.0585** |

Evaluation settings:

```text
Confidence threshold: 0.35
IoU threshold: 0.50
```

The model showed stronger performance on visible and more frequent classes such as:

- `Person`
- `helmet`
- `gloves`

Performance was weaker for several rare missing-PPE classes, including:

- `no_boots`
- `no_gloves`
- `no_goggle`
- `no_helmet`

This is consistent with the short training duration and class imbalance in the dataset.

The evaluation metrics are saved in:

```text
artifacts/evaluation_metrics.json
```

Ultralytics validation outputs and plots are saved under:

```text
runs/siteguard/evaluation/
```

---

## Error Analysis

The confusion matrix shows that the model performs better on common PPE and person classes than on rare violation classes.

Important observations:

- `Person` achieved relatively strong detection performance.
- `helmet` and `gloves` were detected more reliably than several missing-PPE classes.
- Rare classes produced more false negatives.
- Class imbalance affects the model's ability to learn underrepresented violations.
- The short 3-epoch training run limits convergence.

### Possible Improvements

Future improvements may include:

- Training for more epochs
- Applying class balancing
- Collecting more examples of missing-PPE classes
- Tuning confidence and IoU thresholds
- Testing additional augmentation strategies
- Fine-tuning learning-rate settings

---

## Project Structure

```text
SiteGuard-Vision/
│
├── README.md
├── requirements.txt
├── app.py
├── .gitignore
│
├── src/
│   ├── train.py
│   ├── evaluate.py
│   ├── analyze_training.py
│   ├── inference.py
│   ├── segmentation_demo.py
│   ├── video_analytics.py
│   └── export_model.py
│
├── notebooks/
│   └── siteguard_capstone.ipynb
│
├── docs/
│   ├── RUBRIC_CHECKLIST.md
│   ├── EXECUTION_EVIDENCE.md
│   └── TECHNICAL_DOCUMENTATION.md
│
├── artifacts/
│   ├── evaluation_metrics.json
│   └── training_analysis.md
│
└── runs/
    └── siteguard/
```

---

## Installation

Clone the repository:

```bash
git clone <YOUR_REPOSITORY_URL>
cd SiteGuard-Vision
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

If Ultralytics is not installed:

```bash
pip install ultralytics
```

---

## Training

Run custom training:

```bash
python src/train.py
```

The best model weights are saved under the training output directory.

---

## Evaluation

Run model evaluation:

```bash
python src/evaluate.py
```

The script performs validation and saves the evaluation metrics.

---

## Training Analysis

Run:

```bash
python src/analyze_training.py
```

The script analyzes training results and generates a training summary.

---

## Image Inference

Run:

```bash
python src/inference.py
```

The inference pipeline performs YOLO object detection and generates annotated predictions.

---

## Segmentation

Run:

```bash
python src/segmentation_demo.py
```

This executes the segmentation component and generates segmentation-mask visualizations.

---

## Video Analytics

Run:

```bash
python src/video_analytics.py
```

The video pipeline performs detection and tracking on video frames.

The video analytics workflow can visualize:

- Workers
- PPE classes
- Bounding boxes
- Tracking IDs

A YOLO tracking workflow can also be executed using ByteTrack:

```python
results = model.track(
    frame,
    persist=True,
    conf=0.20,
    iou=0.50,
    tracker="bytetrack.yaml",
    verbose=False
)
```

---

## ONNX Export

Run:

```bash
python src/export_model.py
```

The model is exported to ONNX format for deployment.

---

## Deployment

A Streamlit application is included in:

```text
app.py
```

Run the application using:

```bash
streamlit run app.py
```

---

## Execution Evidence

The complete workflow was executed successfully, including:

- Custom YOLO training
- Validation with `model.val()`
- Precision, Recall, mAP@50, and mAP@50-95 reporting
- Confusion-matrix generation
- Image inference
- Segmentation inference
- Video detection and tracking
- ONNX model export

Generated results should be retained in the repository where file-size limits permit.

Recommended evidence includes:

```text
artifacts/evaluation_metrics.json
artifacts/training_analysis.md
runs/siteguard/evaluation/
runs/siteguard/ppe_train/
```

Screenshots or representative output images may also be stored in a dedicated `results/` directory.

---

## Technologies

- Python
- Ultralytics YOLO
- PyTorch
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Streamlit
- ONNX
- Google Colab

---

## Limitations

The current trained model was fine-tuned for only three epochs to complete the full end-to-end workflow within the available training environment.

As a result:

- Recall remains limited.
- Rare PPE-violation classes are underrepresented.
- Additional training is expected to improve detection quality.
- Results should be interpreted as a functional capstone implementation rather than a production-ready safety system.

---

## Future Work

Potential future improvements include:

- Longer training
- Larger custom construction-safety datasets
- Better class balancing
- Restricted-zone event logging
- Real-time camera integration
- PPE violation dashboards
- Alert notifications
- Cloud deployment
- Model optimization for edge devices

---

## Acknowledgment

This project was developed as a computer vision capstone project using **Ultralytics YOLO** and was completed as part of the **SDAIA Academy training program**.

---

## Conclusion

SiteGuard Vision demonstrates a complete computer vision workflow from model training to deployment-oriented export.

The solution integrates:

- Object detection
- Custom training
- Evaluation
- Error analysis
- Segmentation
- Video analytics
- Object tracking
- ONNX export

The project demonstrates how modern computer vision techniques can be applied to construction-site safety monitoring and PPE compliance.
