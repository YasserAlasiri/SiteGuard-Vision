"""Convenience orchestrator for the scored training/evaluation/export stages."""
import subprocess
import sys

steps = [
    [sys.executable, "src/train.py"],
    [sys.executable, "src/evaluate.py"],
    [sys.executable, "src/analyze_training.py"],
    [sys.executable, "src/export_model.py"],
]
for step in steps:
    print("\n=== RUNNING:", " ".join(step), "===")
    subprocess.run(step, check=True)
print("\nCore pipeline complete. Run segmentation_demo.py and video_analytics.py with real media to capture Deliverables 1-2 evidence.")
