"""Turn the real Ultralytics results.csv into a short over/underfitting discussion."""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def find_col(df: pd.DataFrame, includes: tuple[str, ...]):
    for c in df.columns:
        low = c.lower().replace(" ", "")
        if all(k in low for k in includes):
            return c
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="runs/siteguard/ppe_train/results.csv")
    p.add_argument("--output", default="artifacts/training_analysis.md")
    args = p.parse_args()
    df = pd.read_csv(args.csv)
    if df.empty:
        raise ValueError("Training results CSV is empty")

    map_col = find_col(df, ("map50-95",)) or find_col(df, ("map", "b"))
    val_box = find_col(df, ("val/box_loss",))
    train_box = find_col(df, ("train/box_loss",))

    best_epoch = int(df[map_col].idxmax() + 1) if map_col else None
    best_map = float(df[map_col].max()) if map_col else None
    final_map = float(df[map_col].iloc[-1]) if map_col else None

    status = "No automatic conclusion; inspect curves."
    if map_col and len(df) >= 6:
        recent = df[map_col].tail(5)
        if recent.iloc[-1] < recent.max() - 0.02:
            status = "Possible late-stage overfitting: validation mAP peaked earlier and then dropped. Keep best.pt / consider earlier stopping or stronger augmentation."
        elif recent.iloc[-1] >= recent.iloc[0]:
            status = "Validation mAP is stable or improving late in training; no strong overfitting signal from mAP alone."

    lines = [
        "# Training Run Analysis",
        "",
        f"- Epochs completed: {len(df)}",
        f"- Best validation mAP50-95 epoch: {best_epoch}",
        f"- Best validation mAP50-95: {best_map}",
        f"- Final validation mAP50-95: {final_map}",
        f"- Assessment: {status}",
        "",
        "## Hyperparameter rationale",
        "- Image size 640 balances PPE detail and training cost.",
        "- Early stopping patience=12 protects against unnecessary over-training.",
        "- Pretrained weights improve convergence on the 1,416-image domain dataset.",
        "- Default Ultralytics augmentations are retained first; if validation stalls, tune augmentation and learning rate based on the curves rather than guessing.",
    ]
    if train_box and val_box:
        lines += ["", f"Final train box loss: {df[train_box].iloc[-1]}", f"Final val box loss: {df[val_box].iloc[-1]}"]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
