from __future__ import annotations

import numpy as np
import pandas as pd

from config import PROCESSED_DIR, OUTPUTS_DIR, THRESHOLD_START, THRESHOLD_END, THRESHOLD_STEP


SCORES_CSV = PROCESSED_DIR / "lfw_scores_arcface.csv"
THRESHOLD_OUTPUT_CSV = OUTPUTS_DIR / "tables" / "threshold_metrics_arcface.csv"
GROUP_OUTPUT_CSV = OUTPUTS_DIR / "tables" / "group_metrics_arcface.csv"


def compute_confusion(y_true: np.ndarray, y_pred: np.ndarray):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tp, tn, fp, fn


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den != 0 else 0.0


def main() -> None:
    df = pd.read_csv(SCORES_CSV).dropna(subset=["score"]).copy()

    thresholds = np.arange(THRESHOLD_START, THRESHOLD_END + THRESHOLD_STEP, THRESHOLD_STEP)
    summary_rows = []
    group_rows = []

    for tau in thresholds:
        df["pred"] = (df["score"] >= tau).astype(int)

        group_metrics = []

        for group, gdf in df.groupby("group"):
            y_true = gdf["label"].to_numpy()
            y_pred = gdf["pred"].to_numpy()

            tp, tn, fp, fn = compute_confusion(y_true, y_pred)

            acc = safe_div(tp + tn, tp + tn + fp + fn)
            fpr = safe_div(fp, fp + tn)
            fnr = safe_div(fn, fn + tp)

            row = {
                "threshold": tau,
                "group": group,
                "accuracy": acc,
                "fpr": fpr,
                "fnr": fnr,
            }

            group_metrics.append(row)
            group_rows.append(row)

        gm = pd.DataFrame(group_metrics)

        acc_disp = gm["accuracy"].max() - gm["accuracy"].min()
        fpr_disp = gm["fpr"].max() - gm["fpr"].min()
        fnr_disp = gm["fnr"].max() - gm["fnr"].min()
        worst_acc = gm["accuracy"].min()

        summary_rows.append(
            {
                "threshold": tau,
                "accuracy_disparity": acc_disp,
                "fpr_disparity": fpr_disp,
                "fnr_disparity": fnr_disp,
                "worst_group_accuracy": worst_acc,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    group_df = pd.DataFrame(group_rows)

    THRESHOLD_OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(THRESHOLD_OUTPUT_CSV, index=False)
    group_df.to_csv(GROUP_OUTPUT_CSV, index=False)

    print(f"Saved threshold metrics to: {THRESHOLD_OUTPUT_CSV}")
    print(f"Saved group metrics to: {GROUP_OUTPUT_CSV}")


if __name__ == "__main__":
    main()