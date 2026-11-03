#!/usr/bin/env python3
"""Compare cyclone and non-cyclone windows with Cohen's d.

Windows are classified using the same Best Tracks date overlap rule as
``quantitative_analysis.py``: more than five active days is a cyclone window.
The script writes feature-level summary statistics, a sorted Cohen's d bar
plot, and the quantitative analysis scatter plot with the same styling.

Run with the repository's ``dl`` environment, for example:
    python Plots/cohensd.py
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quantitative_analysis import (
    DEFAULT_REGIONS,
    load_cyclone_metrics,
)


FEATURES = [
    "mean_degree",
    "density",
    "n_components",
    "largest_component_nodes",
    "largest_component_fraction",
    "diameter_largest_component",
    "mean_shortest_path_largest_component",
    "mean_betweenness",
    "degree_assortativity",
    "clustering_corr_mean",
    "mean_dist_corr_mean"
]


def summarize_effect_sizes(df: pd.DataFrame) -> pd.DataFrame:
    """Return class sample sizes, means, SDs, and pooled-SD Cohen's d."""
    rows = []
    for feature in FEATURES:
        if feature not in df.columns:
            cyclone = pd.Series(dtype=float)
            non_cyclone = pd.Series(dtype=float)
        else:
            values = pd.to_numeric(df[feature], errors="coerce")
            cyclone = values[df["is_cyclone_active"]].dropna()
            non_cyclone = values[~df["is_cyclone_active"]].dropna()

        n_cyclone, n_non_cyclone = len(cyclone), len(non_cyclone)
        mean_cyclone = cyclone.mean() if n_cyclone else np.nan
        mean_non_cyclone = non_cyclone.mean() if n_non_cyclone else np.nan
        sd_cyclone = cyclone.std(ddof=1) if n_cyclone > 1 else np.nan
        sd_non_cyclone = non_cyclone.std(ddof=1) if n_non_cyclone > 1 else np.nan

        denom_df = n_cyclone + n_non_cyclone - 2
        if denom_df > 0 and pd.notna(sd_cyclone) and pd.notna(sd_non_cyclone):
            pooled_sd = np.sqrt(
                ((n_cyclone - 1) * sd_cyclone**2
                 + (n_non_cyclone - 1) * sd_non_cyclone**2)
                / denom_df
            )
            effect = (mean_cyclone - mean_non_cyclone) / pooled_sd if pooled_sd else np.nan
        else:
            effect = np.nan

        rows.append({
            "feature": feature,
            "n_cyclone": n_cyclone,
            "mean_cyclone": mean_cyclone,
            "std_cyclone": sd_cyclone,
            "n_non_cyclone": n_non_cyclone,
            "mean_non_cyclone": mean_non_cyclone,
            "std_non_cyclone": sd_non_cyclone,
            "cohens_d": abs(effect) if pd.notna(effect) else np.nan,
        })

    return pd.DataFrame(rows).sort_values("cohens_d", ascending=False, na_position="last")


def plot_cohens_d(summary: pd.DataFrame, output_path: Path, dpi: int = 300) -> Path:
    """Save a horizontal bar chart ordered by descending absolute Cohen's d."""
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plotted = summary.dropna(subset=["cohens_d"]).sort_values("cohens_d")

    fig_height = max(5.5, 0.55 * len(plotted) + 1.5)
    fig, ax = plt.subplots(figsize=(10.5, fig_height), dpi=dpi)
    ax.barh(plotted["feature"], plotted["cohens_d"], color="#2878b5", edgecolor="#2c3e50", linewidth=0.5)
    ax.set_xlabel("Absolute Cohen's d, |d|", fontsize=13, fontweight="bold")
    ax.set_ylabel("Feature", fontsize=13, fontweight="bold")
    ax.set_title("Cyclone vs. Non-cyclone Window Effect Sizes", fontsize=16, fontweight="bold", pad=12)
    ax.grid(axis="x", linestyle=":", alpha=0.55, color="#bdc3c7")
    ax.tick_params(axis="both", labelsize=11)
    fig.tight_layout()

    if output_path.suffix.lower() == ".pdf":
        fig.savefig(output_path, format="pdf", bbox_inches="tight", facecolor="white")
        fig.savefig(output_path.with_suffix(".png"), dpi=dpi, bbox_inches="tight", facecolor="white")
    else:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        fig.savefig(output_path.with_suffix(".pdf"), format="pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def plot_top_feature_scatter(
    df: pd.DataFrame,
    features: list[str],
    output_path: Path,
    show: bool = False,
    dpi: int = 300,
) -> Path:
    """Plot the two largest-|d| features using the existing scatter styling."""
    x_feature, y_feature = features
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cyclones = sorted(df["cyclone"].dropna().unique())
    n_cyclones = len(cyclones)
    colors = (plt.cm.tab10(np.linspace(0, 1, 10))[:n_cyclones] if n_cyclones <= 10
              else plt.cm.tab20(np.linspace(0, 1, n_cyclones)))
    cyclone_color_map = {name: colors[i] for i, name in enumerate(cyclones)}

    fig, ax = plt.subplots(figsize=(10.5, 7.0), dpi=dpi)
    for name in cyclones:
        group = df[df["cyclone"] == name]
        inactive = group[~group["is_cyclone_active"]]
        if not inactive.empty:
            ax.scatter(inactive[x_feature], inactive[y_feature], color=cyclone_color_map[name],
                       marker="o", s=55, alpha=0.20, edgecolors="none", zorder=2)
    for name in cyclones:
        group = df[df["cyclone"] == name]
        active = group[group["is_cyclone_active"]]
        if not active.empty:
            ax.scatter(active[x_feature], active[y_feature], color=cyclone_color_map[name],
                       marker="o", s=75, alpha=0.95, edgecolors="#2c3e50",
                       linewidth=0.8, zorder=4)

    ax.set_xlabel(x_feature.replace("_", " ").title(), fontsize=15, fontweight="bold", labelpad=10)
    ax.set_ylabel(y_feature.replace("_", " ").title(), fontsize=15, fontweight="bold", labelpad=10)
    ax.set_title("Top Two Features by Absolute Cohen's d: Bay of Bengal",
                 fontsize=17, fontweight="bold", pad=16)
    ax.grid(True, linestyle=":", alpha=0.55, color="#bdc3c7", zorder=0)
    ax.tick_params(axis="both", which="major", labelsize=13)
    ax.text(0.03, 0.045,
            r"Transparency indicates events that are not cyclones ($\leq 5$ days)",
            transform=ax.transAxes, fontsize=12.5, verticalalignment="bottom",
            horizontalalignment="left",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                      edgecolor="#bdc3c7", alpha=0.92), zorder=5)
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=cyclone_color_map[name],
                   markersize=9.5, markeredgecolor="#2c3e50", markeredgewidth=0.8,
                   alpha=0.95, label=name)
        for name in cyclones
    ]
    ax.legend(handles=legend_handles, bbox_to_anchor=(1.02, 1.0), loc="upper left",
              borderaxespad=0.0, frameon=True, framealpha=0.95,
              edgecolor="#bdc3c7", fontsize=12, title="Cyclones", title_fontsize=13.5)
    fig.tight_layout()
    if output_path.suffix.lower() == ".pdf":
        fig.savefig(output_path, format="pdf", bbox_inches="tight", facecolor="white")
        fig.savefig(output_path.with_suffix(".png"), dpi=dpi, bbox_inches="tight", facecolor="white")
    else:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        fig.savefig(output_path.with_suffix(".pdf"), format="pdf", bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", help="Path to Results directory (default: repository Results/).")
    parser.add_argument("--best-tracks", help="Path to Best Tracks Excel file.")
    parser.add_argument("--max-non-cyclone-days", type=int, default=5,
                        help="Windows with more than this many active days are cyclone windows (default: 5).")
    parser.add_argument("--output-dir", help="Output directory (default: Plots/).")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--show", action="store_true", help="Display the figures interactively.")
    args = parser.parse_args()

    plot_dir = Path(args.output_dir).resolve() if args.output_dir else Path(__file__).resolve().parent
    plot_dir.mkdir(parents=True, exist_ok=True)

    df = load_cyclone_metrics(
        results_dir=Path(args.results_dir).resolve() if args.results_dir else None,
        best_tracks_path=Path(args.best_tracks).resolve() if args.best_tracks else None,
        regions=DEFAULT_REGIONS,
        max_non_cyclone_days=args.max_non_cyclone_days,
    )
    if not df["is_cyclone_active"].any():
        raise RuntimeError(
            "No cyclone windows were classified, so Cohen's d cannot be computed. "
            "Check that openpyxl is installed in the active environment and that "
            "the Best Tracks workbook is being read successfully."
        )
    summary = summarize_effect_sizes(df)
    ranked_features = summary.dropna(subset=["cohens_d"]).head(2)["feature"].tolist()
    if len(ranked_features) < 2:
        raise RuntimeError("At least two features need defined Cohen's d values to make the scatter plot.")
    csv_path = plot_dir / "cohensd_summary.csv"
    summary.to_csv(csv_path, index=False)
    bar_path = plot_cohens_d(summary, plot_dir / "cohensd.pdf", dpi=args.dpi)
    scatter_path = plot_top_feature_scatter(
        df=df,
        features=ranked_features,
        output_path=plot_dir / "cohensd_scatter.pdf",
        show=args.show,
        dpi=args.dpi,
    )
    # The existing scatter routine uses ``cyclone`` for storm names, so add
    # the requested boolean class column after rendering that plot.
    df["cyclone"] = df["is_cyclone_active"].astype(bool)
    classified_path = plot_dir / "cohensd_window_metrics.csv"
    df.to_csv(classified_path, index=False)
    print(f"Summary statistics saved to: {csv_path}")
    print(f"Window data with cyclone class saved to: {classified_path}")
    print(f"Cohen's d bar plot saved to: {bar_path}")
    print(f"Top features by |d|: {ranked_features[0]} vs. {ranked_features[1]}")
    print(f"Top-feature scatter plot saved to: {scatter_path}")
if __name__ == "__main__":
    main()
