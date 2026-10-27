from pathlib import Path
import re

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "Results"

st.set_page_config(
    page_title="Cyclone Dashboard",
    page_icon="🌪️",
    layout="wide",
)

st.title("Cyclone Network Metrics Viewer")
st.caption("Explore the evolution of metrics by window")

GLOBAL_METRICS = [
    "n_nodes",
    "n_edges",
    "mean_degree",
    "degree_std",
    "density",
    "n_components",
    "largest_component_nodes",
    "largest_component_fraction",
    "diameter_largest_component",
    "mean_shortest_path_largest_component",
    "mean_betweenness",
    "max_betweenness",
    "degree_assortativity",
]

CORRECTED_METRICS = ["degree_corr", "mean_dist_corr", "clustering_corr"]
STATISTICS = [
    "mean",
    "median",
    "std",
    "q25",
    "q75",
    "iqr",
    "p10",
    "p90",
    "min",
    "max",
    "skewness",
    "proportion_above_1",
]


def read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [str(col).strip().replace("**", "") for col in df.columns]
    return df


def natural_key(value):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(value))
    ]


def find_available_data(results_root: Path):
    available = []
    if not results_root.exists():
        return available

    for region_dir in sorted(p for p in results_root.iterdir() if p.is_dir()):
        for cyclone_dir in sorted(p for p in region_dir.iterdir() if p.is_dir()):
            window_csv = cyclone_dir / "window_metrics.csv"
            if window_csv.exists():
                available.append(
                    {
                        "region": region_dir.name,
                        "cyclone": cyclone_dir.name,
                        "folder": cyclone_dir,
                        "window_csv": window_csv,
                    }
                )
    return available


def sort_windows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "window" in df.columns:
        numeric_window = pd.to_numeric(df["window"], errors="coerce")
        if numeric_window.notna().any():
            df["_sort_window"] = numeric_window
            df = df.sort_values("_sort_window")
            return df.drop(columns="_sort_window")
        df["_sort_window"] = df["window"].astype(str)
        return df.sort_values("_sort_window", key=lambda s: s.map(natural_key)).drop(
            columns="_sort_window"
        )
    if "start" in df.columns:
        return df.sort_values("start")
    return df


def wide_to_long(window_df: pd.DataFrame):

    id_cols = [c for c in ["window", "start", "end"] if c in window_df.columns]
    stat_names = STATISTICS + ["n_nodes_valid"]

    frames = []
    for metric in CORRECTED_METRICS:
        rename = {
            f"{metric}_{stat}": stat
            for stat in stat_names
            if f"{metric}_{stat}" in window_df.columns
        }
        if not rename:
            continue
        part = window_df[id_cols + list(rename)].rename(columns=rename)
        part.insert(len(id_cols), "metric", metric)
        frames.append(part)

    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def choose_x(df: pd.DataFrame):
    if "start" in df.columns and df["start"].notna().any():
        start = df["start"].astype(str)
        if start.str.strip().ne("").any():
            return "start", "Início da janela"
    if "window" in df.columns:
        return "window", "Janela"
    return None, None


def metric_label(name: str) -> str:
    labels = {
        "mean_degree": "Mean degree",
        "degree_std": "Degree standard deviation",
        "density": "Density",
        "n_nodes": "Number of nodes",
        "n_edges": "Number of edges",
        "n_components": "Number of conected components",
        "largest_component_nodes": "nodes in the largest component",
        "largest_component_fraction": "Fraction of nodes in the largest component",
        "diameter_largest_component": "Diameter of the largest component",
        "mean_shortest_path_largest_component": "Average minimum path (largest component)",
        "mean_betweenness": "Mean betweenness",
        "max_betweenness": "Maximum betweenness",
        "degree_assortativity": "Degree assortativity",
        "degree_corr": "Corrected degree",
        "mean_dist_corr": "Corrected mean geographical distance",
        "clustering_corr": "Corrected local clustering coefficient",
        "mean": "Mean",
        "median": "Median",
        "std": "Standard deviation",
        "q25": "Q25",
        "q75": "Q75",
        "iqr": "Interquartile range (IQR)",
        "p10": "Percentile 10",
        "p90": "Percentile 90",
        "min": "Minimum",
        "max": "Maximum",
        "skewness": "Skewness",
        "proportion_above_1": "Proportion above 1",
        "n_nodes_valid": "Valid nodes",
    }
    return labels.get(name, name.replace("_", " ").capitalize())


def add_reference_line(fig, metric: str):
    if metric in {"degree_corr", "mean_dist_corr", "clustering_corr"}:
        fig.add_hline(
            y=1,
            line_dash="dash",
            annotation_text="Reference = 1",
            annotation_position="top left",
        )
    return fig


available = find_available_data(RESULTS_ROOT)

if not available:
    st.error(
        f"CSV files not found in  `{RESULTS_ROOT}`. "
        "Check to see if the “Results” folder is in the root of the repository and contains "
        "`<região>/<ciclone>/window_metrics.csv`."
    )
    st.code(
        "your_repo/\n"
        "├── Results/\n"
        "│   └── Bengal_Bay/\n"
        "│       └── Gaja/\n"
        "│           └── window_metrics.csv\n"
        "└── dashboard/\n"
        "    └── app.py"
    )
    st.stop()

with st.sidebar:
    st.header("Selection")
    regions = sorted({item["region"] for item in available})
    selected_region = st.selectbox("Region", regions)

    region_items = [item for item in available if item["region"] == selected_region]
    cyclones = sorted({item["cyclone"] for item in region_items})
    selected_cyclone = st.selectbox("Cyclone", cyclones)

selected = next(
    item for item in available
    if item["region"] == selected_region and item["cyclone"] == selected_cyclone
)

try:
    window_df = sort_windows(read_csv(selected["window_csv"]))
except Exception as exc:
    st.error(f"Error reading {selected['window_csv'].name}: {exc}")
    st.stop()

distribution_df = wide_to_long(window_df)

st.subheader(f"{selected_region} · {selected_cyclone}")
c1, c2, c3 = st.columns(3)
c1.metric("Windows", len(window_df))
if "n_nodes" in window_df.columns:
    c2.metric("nodes (first window)", str(window_df["n_nodes"].iloc[0]))
if "n_edges" in window_df.columns:
    c3.metric("Edges (first window)", str(window_df["n_edges"].iloc[0]))

tab_global, tab_corrected, tab_compare, tab_data = st.tabs(
    ["Global Metrics", "Corrected metrics", "Compare statistics", "CSV Data"]
)

with tab_global:
    options = [m for m in GLOBAL_METRICS if m in window_df.columns]
    if not options:
        st.warning("None of the expected global columns were found.")
    else:
        selected_global = st.selectbox(
            "Global metric",
            options,
            format_func=metric_label,
            key="global_metric",
        )
        x_col, x_label = choose_x(window_df)
        if x_col is None:
            st.warning("There is no “window” or “start” column to set up the time axis.")
        else:
            plot_df = window_df.copy()
            plot_df[selected_global] = pd.to_numeric(
                plot_df[selected_global], errors="coerce"
            )
            plot_df = plot_df.dropna(subset=[selected_global])
            fig = px.line(
                plot_df,
                x=x_col,
                y=selected_global,
                markers=True,
                title=f"{metric_label(selected_global)} per window",
                labels={
                    x_col: x_label,
                    selected_global: metric_label(selected_global),
                },
                hover_data=[
                    col for col in ["window", "start", "end"]
                    if col in plot_df.columns and col != x_col
                ],
            )
            fig.update_layout(hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Diameter and average minimum path are calculated for the largest connected "
                "component, depending on the implementation that generated the CSV."
            )

with tab_corrected:
    if distribution_df is None:
        st.info(
            "No corrected metric column (degree_corr_*, mean_dist_corr_*, "
            "clustering_corr_*) found in window_metrics.csv."
        )
    else:
        metrics_available = sorted(distribution_df["metric"].unique())
        selected_metric = st.selectbox(
            "Corrected metric",
            metrics_available,
            format_func=metric_label,
            key="corrected_metric",
        )
        stats_available = [
            stat for stat in STATISTICS
            if stat in distribution_df.columns
        ]
        selected_stat = st.selectbox(
            "Statistic",
            stats_available,
            format_func=metric_label,
            index=stats_available.index("median") if "median" in stats_available else 0,
            key="corrected_stat",
        )
        metric_df = distribution_df[
            distribution_df["metric"].astype(str) == selected_metric
        ].copy()
        metric_df[selected_stat] = pd.to_numeric(
            metric_df[selected_stat], errors="coerce"
        )
        x_col, x_label = choose_x(metric_df)
        if x_col is None:
            x_col, x_label = "window", "Window"
        fig = px.line(
            metric_df,
            x=x_col,
            y=selected_stat,
            markers=True,
            title=f"{metric_label(selected_metric)} — {metric_label(selected_stat)}",
            labels={x_col: x_label, selected_stat: metric_label(selected_stat)},
            hover_data=[
                col for col in ["window", "start", "end", "n_nodes_valid"]
                if col in metric_df.columns and col != x_col
            ],
        )
        add_reference_line(fig, selected_metric)
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Median and interquartile range")
        if {"median", "q25", "q75"}.issubset(metric_df.columns):
            band_df = metric_df.copy()
            for col in ["median", "q25", "q75"]:
                band_df[col] = pd.to_numeric(band_df[col], errors="coerce")
            band_df = band_df.dropna(subset=["median", "q25", "q75"])
            if not band_df.empty:
                band_fig = go.Figure()
                band_fig.add_trace(
                    go.Scatter(
                        x=band_df[x_col],
                        y=band_df["q75"],
                        mode="lines",
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo="skip",
                        name="Q75",
                    )
                )
                band_fig.add_trace(
                    go.Scatter(
                        x=band_df[x_col],
                        y=band_df["q25"],
                        mode="lines",
                        line=dict(width=0),
                        fill="tonexty",
                        fillcolor="rgba(80, 140, 200, 0.20)",
                        name="Q25–Q75",
                        hovertemplate="Window: %{x}<br>Q25: %{y}<extra></extra>",
                    )
                )
                band_fig.add_trace(
                    go.Scatter(
                        x=band_df[x_col],
                        y=band_df["median"],
                        mode="lines+markers",
                        name="Mediana",
                        hovertemplate="Window: %{x}<br>Mediana: %{y}<extra></extra>",
                    )
                )
                add_reference_line(band_fig, selected_metric)
                band_fig.update_layout(
                    title=f"Median and interquartile range — {metric_label(selected_metric)}",
                    xaxis_title=x_label,
                    yaxis_title=metric_label(selected_metric),
                    hovermode="x unified",
                )
                st.plotly_chart(band_fig, use_container_width=True)
        else:
            st.caption("Q25 and Q75 are not available for calculating the interquartile range.")

with tab_compare:
    st.markdown("Compare statistics for more than one adjusted metric in the same window.")
    if distribution_df is None:
        st.info("There are no corrected metrics in window_metrics.csv to compare.")
    else:
        metrics_available = sorted(distribution_df["metric"].unique())
        selected_metrics = st.multiselect(
            "Metrics for comparison",
            metrics_available,
            default=metrics_available[:2],
            format_func=metric_label,
        )
        stats_compare_options = [s for s in STATISTICS if s in distribution_df.columns]
        stat_compare = st.selectbox(
            "Statistics for comparison",
            stats_compare_options,
            index=stats_compare_options.index("median")
            if "median" in stats_compare_options else 0,
            format_func=metric_label,
            key="compare_stat",
        )
        if selected_metrics:
            compare_df = distribution_df[
                distribution_df["metric"].astype(str).isin(selected_metrics)
            ].copy()
            compare_df[stat_compare] = pd.to_numeric(
                compare_df[stat_compare], errors="coerce"
            )
            compare_df["metric_label"] = compare_df["metric"].map(metric_label)
            x_col, x_label = choose_x(compare_df)
            if x_col is None:
                x_col, x_label = "window", "Window"
            fig = px.line(
                compare_df,
                x=x_col,
                y=stat_compare,
                color="metric_label",
                markers=True,
                title=f"Comparison by {metric_label(stat_compare).lower()}",
                labels={
                    x_col: x_label,
                    stat_compare: metric_label(stat_compare),
                    "metric_label": "Metric",
                },
                hover_data=[
                    col for col in ["window", "start", "end"]
                    if col in compare_df.columns and col != x_col
                ],
            )
            fig.update_layout(hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Metrics have different units and interpretations. "
                "Compare their trends with caution; for direct "
                "quantitative comparisons, consider separate scales."
            )
        else:
            st.info("Select at least one metric.")

with tab_data:
    st.markdown("### window_metrics.csv")
    st.dataframe(window_df, use_container_width=True, hide_index=True)
    st.download_button(
        "Download window_metrics.csv",
        data=window_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="window_metrics.csv",
        mime="text/csv",
    )