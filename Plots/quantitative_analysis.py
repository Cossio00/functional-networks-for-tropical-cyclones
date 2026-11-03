#!/usr/bin/env python3
"""
Quantitative Analysis: Clustering Coefficient vs. Mean Degree (Bay of Bengal)
=============================================================================

This script performs quantitative network analysis for tropical cyclones in the
Bay of Bengal by loading windowed functional network metrics from CSV files and
cross-referencing them with cyclone observation dates from the IMD Best Tracks
dataset:
    Dataset/78b4b0_Best_Tracks__Data__1982-2026_.xlsx

Axes:
    - X-axis: Local Clustering Coefficient <C_corr> (column 'clustering_corr_mean')
    - Y-axis: Mean Degree <k> (column 'mean_degree')

Visual Encoding:
    - Points are in FULL COLOR (alpha ~ 0.95, dark borders) when a cyclone event
      is active for at least 5 days within the window ('start' to 'end').
    - Points are FADED (alpha = 0.20, no borders) when fewer than 5 cyclone days
      fall in the window, clearly denoting quiet/background periods.

Data Sources:
    - Results/Bengal_Bay/<Cyclone>/window_metrics.csv
    - Dataset/78b4b0_Best_Tracks__Data__1982-2026_.xlsx

Usage:
    python Plots/quantitative_analysis.py
    python Plots/quantitative_analysis.py --help
    python Plots/quantitative_analysis.py --output Plots/my_figure.png --dpi 300
"""

import argparse
import os
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import linregress, pearsonr, spearmanr


# -----------------------------------------------------------------------------
# Configuration and Constants
# -----------------------------------------------------------------------------

DEFAULT_REGIONS = [
    "Bengal_Bay",
]

REGION_DISPLAY_NAMES = {
    "Bengal_Bay": "Bay of Bengal",
}

DEFAULT_EXCEL_FILENAME = "78b4b0_Best_Tracks__Data__1982-2026_.xlsx"

# Additional known active-date periods. These are merged with Best Tracks
# observations so gaps in the workbook do not cause known cyclone windows to
# be treated as inactive.
ADDITIONAL_CYCLONE_PERIODS = {
    "THANE": (date(2011, 12, 25), date(2011, 12, 31)),
    "HUDHUD": (date(2014, 10, 7), date(2014, 10, 14)),
    "NILOFAR": (date(2014, 10, 25), date(2014, 10, 31)),
    "CHAPALA": (date(2015, 10, 28), date(2015, 11, 4)),
    "MEGH": (date(2015, 11, 5), date(2015, 11, 10)),
    "NADA": (date(2016, 11, 29), date(2016, 12, 2)),
    "VARDAH": (date(2016, 12, 6), date(2016, 12, 13)),
    "OCKHI": (date(2017, 11, 29), date(2017, 12, 5)),
    "DAYE": (date(2018, 9, 19), date(2018, 9, 22)),
    "LUBAN": (date(2018, 10, 6), date(2018, 10, 15)),
    "TITLI": (date(2018, 10, 8), date(2018, 10, 13)),
    "GAJA": (date(2018, 11, 10), date(2018, 11, 19)),
    "PHETHAI": (date(2018, 12, 13), date(2018, 12, 18)),
}


# -----------------------------------------------------------------------------
# Path Resolvers
# -----------------------------------------------------------------------------

def resolve_repo_root() -> Path:
    """Finds repository root relative to this script: Plots/../"""
    return Path(__file__).resolve().parent.parent


def resolve_results_dir(custom_path: Optional[str] = None) -> Path:
    """Locates the Results directory relative to repo root or custom path."""
    if custom_path:
        p = Path(custom_path).resolve()
        if p.exists():
            return p
        raise FileNotFoundError(f"Specified Results directory not found: {custom_path}")

    repo_root = resolve_repo_root()
    candidate = repo_root / "Results"
    if candidate.exists():
        return candidate

    cwd_candidate = Path("Results").resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    raise FileNotFoundError("Could not automatically locate the 'Results' directory.")


def resolve_best_tracks_file(custom_path: Optional[str] = None) -> Path:
    """Locates the Best Tracks Excel file."""
    if custom_path:
        p = Path(custom_path).resolve()
        if p.exists():
            return p
        raise FileNotFoundError(f"Specified Best Tracks file not found: {custom_path}")

    repo_root = resolve_repo_root()
    candidate = repo_root / "Dataset" / DEFAULT_EXCEL_FILENAME
    if candidate.exists():
        return candidate

    cwd_candidate = Path("Dataset") / DEFAULT_EXCEL_FILENAME
    if cwd_candidate.exists():
        return cwd_candidate.resolve()

    raise FileNotFoundError(
        f"Could not locate Best Tracks Excel file: Dataset/{DEFAULT_EXCEL_FILENAME}"
    )


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strips whitespace and markdown artifacts from column names."""
    df.columns = [str(col).strip().replace("**", "") for col in df.columns]
    return df


# -----------------------------------------------------------------------------
# Cyclone Observation Dates Loading
# -----------------------------------------------------------------------------

def load_cyclone_active_dates(
    excel_path: Optional[Path] = None,
    years_filter: Optional[Set[str]] = None,
) -> Dict[str, Set[date]]:
    """
    Parses cyclone observation dates from the IMD Best Tracks Excel file.

    Args:
        excel_path: Path to 78b4b0_Best_Tracks__Data__1982-2026_.xlsx
        years_filter: Optional set of sheet years to read (optimizes performance).

    Returns:
        Dict mapping uppercase cyclone name to set of active datetime.date objects.
    """
    cyclone_dates: Dict[str, Set[date]] = {}

    try:
        excel_file = resolve_best_tracks_file(str(excel_path) if excel_path else None)
    except FileNotFoundError as err:
        print(f"[Warning] {err}. Continuing without external dates.", file=sys.stderr)
        return cyclone_dates

    try:
        xl = pd.ExcelFile(excel_file)
        sheets = xl.sheet_names
        if years_filter:
            sheets = [s for s in sheets if s in years_filter]

        for sheet in sheets:
            df = xl.parse(sheet)
            name_col = next(
                (c for c in df.columns if str(c).strip().lower() == "name"),
                None,
            )
            date_col = next(
                (c for c in df.columns if "date" in str(c).strip().lower()),
                None,
            )
            if not name_col or not date_col:
                continue

            for name, group in df.groupby(df[name_col].astype(str).str.strip().str.upper()):
                dates: Set[date] = set()
                for val in group[date_col].dropna():
                    if isinstance(val, (datetime, pd.Timestamp)):
                        d = val
                    else:
                        try:
                            d = pd.to_datetime(val, dayfirst=True)
                        except Exception:
                            continue

                    # Handle Excel day-month swap for Luban 2018 (11-10 -> 10-11, 12-10 -> 10-12)
                    if name == "LUBAN" and d.year == 2018 and d.month in [11, 12] and d.day == 10:
                        d = d.replace(month=10, day=d.month)

                    dates.add(d.date())

                if dates:
                    cyclone_dates.setdefault(name, set()).update(dates)

    except Exception as e:
        print(f"[Warning] Error parsing {excel_file}: {e}", file=sys.stderr)

    return cyclone_dates


# -----------------------------------------------------------------------------
# Data Loading and Window Cross-Referencing
# -----------------------------------------------------------------------------

def load_cyclone_metrics(
    results_dir: Optional[Path] = None,
    best_tracks_path: Optional[Path] = None,
    regions: Optional[List[str]] = None,
    max_non_cyclone_days: int = 5,
) -> pd.DataFrame:
    """
    Scans Results/Bengal_Bay, loads window_metrics.csv files, and crosses
    each window's start/end dates with cyclone observation dates.

    Args:
        results_dir: Base directory containing region subfolders.
        best_tracks_path: Path to the Best Tracks Excel file.
        regions: List of region folder names to look for (default: ['Bengal_Bay']).
        max_non_cyclone_days: Upper threshold of active days considered non-cyclone (default: 5).
                              Windows with > max_non_cyclone_days are rendered in full color.
                              Windows with <= max_non_cyclone_days are rendered transparent (alpha 0.2).

    Returns:
        pd.DataFrame containing data with columns:
            ['region', 'region_display', 'cyclone', 'window', 'start', 'end',
             'mean_degree', 'clustering_corr_mean', 'active_cyclone_days', 'is_cyclone_active']
    """
    if results_dir is None:
        results_dir = resolve_results_dir()

    if regions is None:
        regions = DEFAULT_REGIONS

    records: List[pd.DataFrame] = []
    seen_files = set()
    years_needed: Set[str] = set()

    for region_name in regions:
        region_path = results_dir / region_name
        if not region_path.is_dir():
            continue

        display_name = REGION_DISPLAY_NAMES.get(region_name, region_name.replace("_", " "))

        for cyclone_dir in sorted(region_path.iterdir()):
            if not cyclone_dir.is_dir():
                continue

            csv_file = cyclone_dir / "window_metrics.csv"
            real_file = csv_file.resolve()

            if not csv_file.exists() or real_file in seen_files:
                continue
            seen_files.add(real_file)

            try:
                df = pd.read_csv(csv_file, encoding="utf-8-sig")
                df = clean_columns(df)

                if "mean_degree" not in df.columns or "clustering_corr_mean" not in df.columns:
                    print(
                        f"[Warning] Skipping {csv_file}: required columns "
                        "('mean_degree', 'clustering_corr_mean') not found.",
                        file=sys.stderr,
                    )
                    continue

                df["mean_degree"] = pd.to_numeric(df["mean_degree"], errors="coerce")
                df["clustering_corr_mean"] = pd.to_numeric(df["clustering_corr_mean"], errors="coerce")

                valid_mask = df["mean_degree"].notna() & df["clustering_corr_mean"].notna()
                df_valid = df.loc[valid_mask].copy()

                if df_valid.empty:
                    print(f"[Warning] No valid data in {csv_file}.", file=sys.stderr)
                    continue

                df_valid["cyclone"] = cyclone_dir.name
                df_valid["region"] = region_name
                df_valid["region_display"] = display_name

                # Collect years for fast Excel parsing
                if "start" in df_valid.columns:
                    for s in df_valid["start"].dropna():
                        try:
                            years_needed.add(str(pd.to_datetime(s).year))
                        except Exception:
                            pass
                if "end" in df_valid.columns:
                    for e in df_valid["end"].dropna():
                        try:
                            years_needed.add(str(pd.to_datetime(e).year))
                        except Exception:
                            pass

                records.append(df_valid)

            except Exception as e:
                print(f"[Error] Failed to read {csv_file}: {e}", file=sys.stderr)

    if not records:
        raise ValueError(
            f"No valid 'window_metrics.csv' files found under {results_dir} "
            f"for regions: {regions}"
        )

    combined_df = pd.concat(records, ignore_index=True)

    # Load cyclone observation dates
    cyclone_active_dates = load_cyclone_active_dates(
        excel_path=best_tracks_path,
        years_filter=years_needed,
    )

    # Merge user-supplied event periods with workbook dates. Sets prevent
    # duplicate days from increasing the overlap count.
    for cyclone_name, (start_date, end_date) in ADDITIONAL_CYCLONE_PERIODS.items():
        period_dates = set(pd.date_range(start_date, end_date).date)
        cyclone_active_dates.setdefault(cyclone_name, set()).update(period_dates)

    # Cross-reference each window with cyclone active dates
    active_days_list: List[int] = []
    is_active_list: List[bool] = []

    for _, row in combined_df.iterrows():
        cname = str(row["cyclone"]).strip().upper()
        active_dates = cyclone_active_dates.get(cname, set())

        if "start" in row and "end" in row and pd.notna(row["start"]) and pd.notna(row["end"]):
            try:
                s_date = pd.to_datetime(row["start"]).date()
                e_date = pd.to_datetime(row["end"]).date()
                w_dates = set(pd.date_range(s_date, e_date).date)
                overlap = len(w_dates.intersection(active_dates))
            except Exception:
                overlap = 0
        else:
            overlap = 0

        active_days_list.append(overlap)
        is_active_list.append(overlap > max_non_cyclone_days)

    combined_df["active_cyclone_days"] = active_days_list
    combined_df["is_cyclone_active"] = is_active_list

    return combined_df


# -----------------------------------------------------------------------------
# Statistical Analysis Helpers
# -----------------------------------------------------------------------------

def compute_statistics(df: pd.DataFrame) -> Dict[str, dict]:
    """Computes correlation and summary statistics."""
    x = df["clustering_corr_mean"].values
    y = df["mean_degree"].values
    p_r, p_val = pearsonr(x, y)
    s_rho, s_val = spearmanr(x, y)
    slope, intercept, r_value, _, stderr = linregress(x, y)

    return {
        "n_total": len(df),
        "n_active": int(df["is_cyclone_active"].sum()),
        "n_inactive": int((~df["is_cyclone_active"]).sum()),
        "cyclones": sorted(df["cyclone"].unique().tolist()),
        "pearson_r": p_r,
        "pearson_p": p_val,
        "spearman_rho": s_rho,
        "spearman_p": s_val,
        "slope": slope,
        "intercept": intercept,
        "r2": r_value**2,
        "stderr": stderr,
    }


def print_summary_report(df: pd.DataFrame, stats: Dict[str, dict]) -> None:
    """Prints a structured summary of loaded data and correlations to terminal."""
    print("\n" + "=" * 70)
    print("QUANTITATIVE ANALYSIS: CLUSTERING COEFFICIENT vs. MEAN DEGREE")
    print("REGION: BAY OF BENGAL")
    print("=" * 70)
    print(f"Total data points (windows): {stats['n_total']}")
    print(f"  • Cyclone event windows (> 5 days): {stats['n_active']} (Full Color)")
    print(f"  • Non-cyclone windows (≤ 5 days): {stats['n_inactive']} (Transparent, alpha 0.2)")
    print(f"Cyclones analyzed ({len(stats['cyclones'])}): {', '.join(stats['cyclones'])}")
    print("-" * 70)
    print(
        f"Pearson correlation:  r = {stats['pearson_r']:+.4f} "
        f"(p = {stats['pearson_p']:.2e}, R² = {stats['r2']:.4f})"
    )
    print(
        f"Spearman correlation: ρ = {stats['spearman_rho']:+.4f} "
        f"(p = {stats['spearman_p']:.2e})"
    )
    print("=" * 70 + "\n")


# -----------------------------------------------------------------------------
# Visualization Functions
# -----------------------------------------------------------------------------

def plot_quantitative_analysis(
    df: pd.DataFrame,
    output_path: Path,
    show: bool = False,
    dpi: int = 300,
) -> Path:
    """
    Creates and saves the primary publication-quality scatter plot of
    Clustering Coefficient (x-axis) vs Mean Degree (y-axis) for Bay of Bengal.

    Points with > 5 cyclone days are rendered in full color.
    Points with <= 5 cyclone days are rendered with alpha = 0.20 (non-cyclone events).

    Args:
        df: DataFrame containing metrics and activity flags.
        output_path: Destination file path for saving the figure (.pdf or .png).
        show: If True, display the interactive window.
        dpi: Resolution of saved raster image.

    Returns:
        Path to the saved figure.
    """
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cyclones = sorted(df["cyclone"].unique())
    n_cyclones = len(cyclones)

    if n_cyclones <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, 10))[:n_cyclones]
    else:
        colors = plt.cm.tab20(np.linspace(0, 1, n_cyclones))
    cyclone_color_map = {c: colors[i] for i, c in enumerate(cyclones)}

    fig, ax = plt.subplots(figsize=(10.5, 7.0), dpi=dpi)

    # 1. Plot inactive / non-cyclone points first (alpha = 0.20) in background
    for cyclone in cyclones:
        sub = df[df["cyclone"] == cyclone]
        inact = sub[~sub["is_cyclone_active"]]

        if not inact.empty:
            ax.scatter(
                inact["clustering_corr_mean"],
                inact["mean_degree"],
                color=cyclone_color_map[cyclone],
                marker="o",
                s=55,
                alpha=0.20,
                edgecolors="none",
                zorder=2,
            )

    # 2. Plot active cyclone points on top in full color (alpha = 0.95, dark border)
    for cyclone in cyclones:
        sub = df[df["cyclone"] == cyclone]
        act = sub[sub["is_cyclone_active"]]

        if not act.empty:
            ax.scatter(
                act["clustering_corr_mean"],
                act["mean_degree"],
                color=cyclone_color_map[cyclone],
                marker="o",
                s=75,
                alpha=0.95,
                edgecolors="#2c3e50",
                linewidth=0.8,
                zorder=4,
            )

    # Axis Labels & Title with enlarged font sizes (increased by >= 2.5 - 3 points)
    ax.set_xlabel(
        r"Local Clustering Coefficient $\langle C_{\mathrm{corr}} \rangle$ (SERN corrected)",
        fontsize=15,
        fontweight="bold",
        labelpad=10,
    )
    ax.set_ylabel(
        r"Mean Degree $\langle k \rangle$ (connections per node)",
        fontsize=15,
        fontweight="bold",
        labelpad=10,
    )
    ax.set_title(
        "Quantitative Network Analysis: Bay of Bengal",
        fontsize=17,
        fontweight="bold",
        pad=16,
    )

    # Grid & Styling
    ax.grid(True, linestyle=":", alpha=0.55, color="#bdc3c7", zorder=0)
    ax.tick_params(axis="both", which="major", labelsize=13)

    # Bottom-left annotation describing transparency for non-cyclone events (<= 5 days)
    annotation_text = r"Transparency indicates events that are not cyclones ($\leq 5$ days)"
    ax.text(
        0.03,
        0.045,
        annotation_text,
        transform=ax.transAxes,
        fontsize=12.5,
        verticalalignment="bottom",
        horizontalalignment="left",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="white",
            edgecolor="#bdc3c7",
            alpha=0.92,
        ),
        zorder=5,
    )

    # Legend with ONLY Cyclone names (Active / Inactive items excluded per user instruction)
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=cyclone_color_map[c],
            markersize=9.5,
            markeredgecolor="#2c3e50",
            markeredgewidth=0.8,
            alpha=0.95,
            label=c,
        )
        for c in cyclones
    ]

    ax.legend(
        handles=legend_handles,
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        borderaxespad=0.0,
        frameon=True,
        framealpha=0.95,
        edgecolor="#bdc3c7",
        fontsize=12,
        title="Cyclones",
        title_fontsize=13.5,
    )

    plt.tight_layout()

    # Save to requested file format
    if output_path.suffix.lower() == ".pdf":
        plt.savefig(output_path, format="pdf", bbox_inches="tight", facecolor="white")
        # Also maintain companion PNG for quick preview
        png_counterpart = output_path.with_suffix(".png")
        plt.savefig(png_counterpart, dpi=dpi, bbox_inches="tight", facecolor="white")
    else:
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        # Also produce companion PDF
        pdf_counterpart = output_path.with_suffix(".pdf")
        plt.savefig(pdf_counterpart, format="pdf", bbox_inches="tight", facecolor="white")

    if show:
        plt.show()

    plt.close(fig)
    return output_path


# -----------------------------------------------------------------------------
# Main Execution Entry Point
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Creates a scatter plot of Clustering Coefficient (x-axis) vs. "
            "Mean Degree (y-axis) for Bay of Bengal tropical cyclones, "
            "cross-referencing observation dates from Best Tracks data."
        )
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Path to Results directory (default: auto-detected repo Results/).",
    )
    parser.add_argument(
        "--best-tracks",
        type=str,
        default=None,
        help="Path to Best Tracks Excel file (default: auto-detected Dataset/78b4b0_Best_Tracks__Data__1982-2026_.xlsx).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the generated scatter plot figure (default: Plots/quantitative_analysis.pdf).",
    )
    parser.add_argument(
        "--max-non-cyclone-days",
        type=int,
        default=5,
        help="Maximum active days considered non-cyclone (<= this value is transparent, default: 5).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolution in DPI for output figures (default: 300).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display interactive plot window before saving.",
    )

    args = parser.parse_args()

    # Determine default output paths (.pdf by default)
    script_dir = Path(__file__).resolve().parent
    if args.output:
        main_output = Path(args.output).resolve()
    else:
        main_output = script_dir / "quantitative_analysis.pdf"

    print("Loading Bay of Bengal cyclone metrics and crossing with Best Tracks data...")
    df = load_cyclone_metrics(
        results_dir=args.results_dir,
        best_tracks_path=args.best_tracks,
        regions=DEFAULT_REGIONS,
        max_non_cyclone_days=args.max_non_cyclone_days,
    )

    # Compute and display statistical summary
    stats = compute_statistics(df)
    print_summary_report(df, stats)

    # Generate primary scatter plot (saves PDF and PNG)
    print(f"Generating scatter plot: {main_output} ...")
    saved_path = plot_quantitative_analysis(
        df=df,
        output_path=main_output,
        show=args.show,
        dpi=args.dpi,
    )
    print(f"✔ PDF successfully saved to: {saved_path}")
    if saved_path.suffix.lower() == ".pdf":
        print(f"✔ PNG preview saved to: {saved_path.with_suffix('.png')}")


if __name__ == "__main__":
    main()
