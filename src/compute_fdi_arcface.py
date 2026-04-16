from __future__ import annotations

import numpy as np
import pandas as pd

from config import OUTPUTS_DIR


INPUT_CSV = OUTPUTS_DIR / "tables" / "threshold_metrics_arcface.csv"
OUTPUT_CSV = OUTPUTS_DIR / "tables" / "fdi_results_arcface.csv"


METRIC_COLUMNS = [
    "accuracy_disparity",
    "fpr_disparity",
    "fnr_disparity",
    "worst_group_accuracy",
]


def minmax_normalize(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        col_min = out[col].min()
        col_max = out[col].max()
        if col_max == col_min:
            out[col] = 0.0
        else:
            out[col] = (out[col] - col_min) / (col_max - col_min)
    return out


def rank_values(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average").to_numpy()


def main() -> None:
    df = pd.read_csv(INPUT_CSV)
    norm_df = minmax_normalize(df, METRIC_COLUMNS)

    rows = []

    for _, row in norm_df.iterrows():
        values = np.array([row[c] for c in METRIC_COLUMNS], dtype=float)
        n = len(values)

        value_disagreements = []
        rank_disagreements = []

        ranks = rank_values(values)

        for i in range(n):
            for j in range(i + 1, n):
                dij = abs(values[i] - values[j])
                rij = abs(ranks[i] - ranks[j])
                value_disagreements.append(dij)
                rank_disagreements.append(rij)

        D = float(np.mean(value_disagreements))
        R = float(np.mean(rank_disagreements))
        alpha = 0.5
        fdi = alpha * D + (1 - alpha) * R

        rows.append(
            {
                "threshold": row["threshold"],
                "value_disagreement": D,
                "rank_disagreement": R,
                "fdi": fdi,
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved FDI results to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()