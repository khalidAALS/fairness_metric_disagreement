from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt

from config import OUTPUTS_DIR


THRESHOLD_CSV = OUTPUTS_DIR / "tables" / "threshold_metrics.csv"
FDI_CSV = OUTPUTS_DIR / "tables" / "fdi_results.csv"
GROUP_CSV = OUTPUTS_DIR / "tables" / "group_metrics_by_threshold.csv"
FIGURES_DIR = OUTPUTS_DIR / "figures"


def plot_metric_comparison_across_groups(threshold: float = 0.50) -> None:
    df = pd.read_csv(GROUP_CSV)

    # picks the closest threshold present in the CSV
    available = sorted(df["threshold"].unique())
    chosen = min(available, key=lambda x: abs(x - threshold))

    subset = df[df["threshold"] == chosen].copy()
    # keeps top 10 groups by frequency
    group_counts = df[df["threshold"] == chosen]["group"].value_counts()
    top_groups = group_counts.head(10).index

    subset = subset[subset["group"].isin(top_groups)]
    subset = subset.sort_values("group")

    plt.figure(figsize=(10, 5))
    plt.plot(subset["group"], subset["accuracy"], label="Accuracy")
    plt.plot(subset["group"], subset["fpr"], label="FPR")
    plt.plot(subset["group"], subset["fnr"], label="FNR")

    plt.xlabel("Group")
    plt.ylabel("Accuracy / Error Rate")
    plt.title(f"Metric Comparison Across Representative Groups (τ = {chosen:.2f})")
    plt.xticks(rotation=90)
    plt.legend()
    plt.tight_layout()

    output_path = FIGURES_DIR / "fig1_metric_comparison_across_groups.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def plot_fairness_metrics_vs_threshold() -> None:
    df = pd.read_csv(THRESHOLD_CSV)

    plt.figure(figsize=(8, 5))
    plt.plot(df["threshold"], df["accuracy_disparity"], label="Accuracy disparity")
    plt.plot(df["threshold"], df["fpr_disparity"], label="FPR disparity")
    plt.plot(df["threshold"], df["fnr_disparity"], label="FNR disparity")
    plt.plot(df["threshold"], df["worst_group_accuracy"], label="Worst-group accuracy")

    plt.xlabel("Threshold")
    plt.ylabel("Disparity / Accuracy Value")
    plt.title("Fairness Metrics Across Decision Thresholds")
    plt.legend()
    plt.tight_layout()

    output_path = FIGURES_DIR / "fig2_fairness_metrics_vs_threshold.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def plot_fdi_vs_threshold() -> None:
    df = pd.read_csv(FDI_CSV)

    plt.figure(figsize=(8, 5))
    plt.plot(df["threshold"], df["fdi"])

    plt.xlabel("Threshold")
    plt.ylabel("FDI")
    plt.title("Fairness Disagreement Index Across Thresholds")
    plt.tight_layout()

    output_path = FIGURES_DIR / "fig3_fdi_vs_threshold.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")
    
def plot_fdi_comparison() -> None:
    import pandas as pd
    import matplotlib.pyplot as plt

    f1 = pd.read_csv(OUTPUTS_DIR / "tables" / "fdi_results.csv")
    f2 = pd.read_csv(OUTPUTS_DIR / "tables" / "fdi_results_arcface.csv")

    plt.figure(figsize=(8, 5))

    plt.plot(f1["threshold"], f1["fdi"], label="FaceNet", linewidth=2)
    plt.plot(f2["threshold"], f2["fdi"], label="ArcFace", linewidth=2)

    plt.xlabel("Threshold")
    plt.ylabel("FDI")
    plt.title("FDI Comparison Across Models")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    output_path = OUTPUTS_DIR / "figures" / "fig4_fdi_comparison.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_metric_comparison_across_groups(threshold=0.50)
    plot_fairness_metrics_vs_threshold()
    plot_fdi_vs_threshold()
    plot_fdi_comparison()


if __name__ == "__main__":
    main()