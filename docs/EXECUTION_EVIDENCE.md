# Execution Evidence

## Training

The YOLO model was trained on the Construction-PPE dataset for 3 epochs.

- Best validation mAP@50-95 epoch: 3
- Best validation mAP@50-95: 0.12953
- Final validation mAP@50-95: 0.12953
- Final training box loss: 1.74261
- Final validation box loss: 1.73273

The best model weights were saved in:

`runs/siteguard/ppe_train/weights/best.pt`

## Evaluation

The trained model was evaluated using Ultralytics `model.val()`.

Final evaluation results:

- Precision: 0.3470
- Recall: 0.1354
- mAP@50: 0.1138
- mAP@50-95: 0.0585

The evaluation metrics were saved in:

`artifacts/evaluation_metrics.json`

## Confusion Matrix

A confusion matrix was generated during validation.

The model showed better performance on common classes such as Person, helmet, gloves, and vest, while performance was weaker on rare missing-PPE classes.

## Image Inference

The trained model was tested on images and generated bounding boxes with class labels.

## Segmentation

A YOLO segmentation model was used as an additional computer vision task beyond object detection.

## Video Analytics

The video pipeline was tested successfully and included:

- Worker detection
- PPE detection
- Bounding boxes
- Object tracking
- Persistent tracking IDs

## Model Export

The trained model was exported to ONNX format for deployment.
