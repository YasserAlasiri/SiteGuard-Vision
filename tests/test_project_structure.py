from pathlib import Path


def test_required_files_exist():
    root = Path(__file__).resolve().parents[1]
    required = [
        "README.md", ".gitignore", "requirements.txt", "src/train.py", "src/evaluate.py",
        "src/inference.py", "src/segmentation_demo.py", "src/video_analytics.py", "src/export_model.py",
        "docs/TECHNICAL_DOCUMENTATION.md", "docs/RUBRIC_CHECKLIST.md"
    ]
    missing = [f for f in required if not (root / f).exists()]
    assert not missing, f"Missing required files: {missing}"
