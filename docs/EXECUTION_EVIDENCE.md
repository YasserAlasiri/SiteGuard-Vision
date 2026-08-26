# Execution Evidence Guide

The rubric explicitly requires **real captured execution**, not code that merely could run. This file tells you exactly what to retain after running the project. Do not replace these items with invented metrics.

1. **Training evidence:** terminal output from `python src/train.py`, `runs/siteguard/ppe_train/results.csv`, `results.png`, and `weights/best.pt`.
2. **Evaluation evidence:** terminal output from `python src/evaluate.py`, `artifacts/evaluation_metrics.json`, and the generated validation plots/confusion matrix under `runs/siteguard/evaluation/`.
3. **Inference evidence:** an annotated result from `python src/inference.py --source <image>`.
4. **Segmentation evidence:** annotated masks from `python src/segmentation_demo.py --source <image>`.
5. **Video analytics evidence:** `artifacts/siteguard_analytics.mp4`, `artifacts/video_analytics_summary.json`, and a screenshot showing track IDs / zone / alert overlays.
6. **Training interpretation:** `artifacts/training_analysis.md` from `python src/analyze_training.py`.
7. **Export evidence:** terminal output from `python src/export_model.py` plus the generated `.onnx` file.
8. **App evidence:** screenshot of `streamlit run app.py` processing a construction image.

For GitHub, keep screenshots or small JSON/Markdown evidence files; do not commit large datasets, model weights, videos, or generated `runs/` unless your instructor explicitly asks for them.
