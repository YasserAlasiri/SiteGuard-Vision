## Training Analysis

The model was trained on the Construction-PPE dataset for 3 epochs using pretrained YOLO weights.

Training summary:

- Best validation mAP@50-95 epoch: 3
- Best validation mAP@50-95: 0.12953
- Final validation mAP@50-95: 0.12953
- Final training box loss: 1.74261
- Final validation box loss: 1.73273

The training and validation box losses remained close during the training run. This does not show a clear sign of overfitting.

However, the training duration was short, so the model likely did not reach full convergence. The current results are more consistent with under-training than overfitting.

## Evaluation Results

The trained model was evaluated using Ultralytics `model.val()`.

- Precision: 0.3470
- Recall: 0.1354
- mAP@50: 0.1138
- mAP@50-95: 0.0585

The model performed better on common classes such as Person, helmet, gloves, and vest.

Performance was weaker on rare classes such as no_boots, no_gloves, no_goggle, and no_helmet.

The lower performance on rare classes is likely related to class imbalance and the short training run.

## Video Analytics

The video pipeline uses YOLO and OpenCV to process video frames.

The system performs object detection and tracking and displays bounding boxes, class labels, and persistent tracking IDs.

## Model Export

The trained model was exported to ONNX format to support deployment and interoperability.
