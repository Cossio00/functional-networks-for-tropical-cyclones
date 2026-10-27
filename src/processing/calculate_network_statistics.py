import os
import pickle
import warnings

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import skew


def summarize_distribution(values):

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return {
            "n_nodes_valid": 0,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "q25": np.nan,
            "q75": np.nan,
            "iqr": np.nan,
            "p10": np.nan,
            "p90": np.nan,
            "min": np.nan,
            "max": np.nan,
            "skewness": np.nan,
            "proportion_above_1": np.nan,
        }

    q25, q75 = np.percentile(values, [25, 75])

    skewness = (
        float(skew(values, bias=False))
        if len(values) >= 3 and np.std(values) > 0
        else np.nan
    )

    return {
        "n_nodes_valid": len(values),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": (
            float(np.std(values, ddof=1))
            if len(values) > 1
            else np.nan
        ),
        "q25": float(q25),
        "q75": float(q75),
        "iqr": float(q75 - q25),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "skewness": skewness,
        "proportion_above_1": float(np.mean(values > 1)),
    }


def calculate_global_metrics(adjacency_matrix):

    adjacency = np.asarray(adjacency_matrix)

    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(
            "The adjacency matrix must be square."
        )

    if not np.all(np.isfinite(adjacency)):
        raise ValueError(
            "The adjacency matrix contains non-finite values."
        )

    if not np.allclose(adjacency, adjacency.T):
        raise ValueError(
            "The adjacency matrix is not symmetric."
        )

    if not np.all(np.isin(adjacency, [0, 1])):
        raise ValueError(
            "The matrix must represent a binary array (0/1)."
        )

    adjacency = adjacency.astype(int)

    if np.any(np.diag(adjacency) != 0):
        raise ValueError(
            "The matrix contains self-connections on the diagonal."
        )

    graph = nx.from_numpy_array(adjacency)

    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()

    degrees = np.array(
        [degree for _, degree in graph.degree()],
        dtype=float,
    )

    components = list(nx.connected_components(graph))
    largest_component = (
        max(components, key=len) if components else set()
    )

    largest_graph = graph.subgraph(largest_component)

    if len(largest_component) >= 2:
        diameter = nx.diameter(largest_graph)
        mean_shortest_path = nx.average_shortest_path_length(
            largest_graph
        )
    else:
        diameter = 0 if len(largest_component) == 1 else np.nan
        mean_shortest_path = np.nan

    betweenness = nx.betweenness_centrality(
        graph,
        normalized=True,
    )

    betweenness_values = np.array(
        list(betweenness.values()),
        dtype=float,
    )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            degree_assortativity = (
                nx.degree_assortativity_coefficient(graph)
            )
    except (ValueError, ZeroDivisionError):
        degree_assortativity = np.nan

    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "mean_degree": (
            float(np.mean(degrees)) if n_nodes else np.nan
        ),
        "degree_std": (
            float(np.std(degrees, ddof=1))
            if n_nodes > 1
            else np.nan
        ),
        "density": float(nx.density(graph)),
        "n_components": len(components),
        "largest_component_nodes": len(largest_component),
        "largest_component_fraction": (
            len(largest_component) / n_nodes
            if n_nodes
            else np.nan
        ),
        "diameter_largest_component": diameter,
        "mean_shortest_path_largest_component": mean_shortest_path,
        "mean_betweenness": (
            float(np.mean(betweenness_values))
            if len(betweenness_values)
            else np.nan
        ),
        "max_betweenness": (
            float(np.max(betweenness_values))
            if len(betweenness_values)
            else np.nan
        ),
        "degree_assortativity": degree_assortativity,
    }


def calculate_network_statistics(region, cyclone):

    input_path = (
        f"Metrics/{region}/{cyclone}/"
        f"{cyclone}_metrics.pkl"
    )

    if not os.path.isfile(input_path):
        raise FileNotFoundError(
            f"Metrics file not found: {input_path}"
        )

    with open(input_path, "rb") as file:
        data = pickle.load(file)

    if not data:
        raise ValueError(
            f"No windows found for {cyclone}."
        )

    output_dir = f"Results/{region}/{cyclone}"
    os.makedirs(output_dir, exist_ok=True)

    window_rows = []

    metric_fields = {
        "degree_corr": "degree_corr",
        "mean_dist_corr": "mean_dist_corr",
        "clustering_corr": "clustering_corr",
    }

    print(
        f"\nCalculating {cyclone} statistics: "
        f"{len(data)} window(s)."
    )

    for window_key in sorted(data.keys()):
        window_data = data[window_key]

        window_number = window_data.get(
            "window", window_key
        )

        row = {
            "window": window_number,
            "start": window_data.get("start", ""),
            "end": window_data.get("end", ""),
        }

        if "adjacency_matrix" not in window_data:
            raise KeyError(
                f"'adjacency_matrix' missing in window {window_key}."
            )

        global_metrics = calculate_global_metrics(
            window_data["adjacency_matrix"]
        )
        row.update(global_metrics)

        for field, metric_name in metric_fields.items():
            if field not in window_data:
                raise KeyError(
                    f"'{field}' missing in window {window_key}."
                )

            values = np.asarray(
                window_data[field],
                dtype=float,
            )

            summary = summarize_distribution(values)

            for stat_name, stat_value in summary.items():
                row[f"{metric_name}_{stat_name}"] = stat_value

        window_rows.append(row)

    window_df = pd.DataFrame(window_rows)

    output_path = os.path.join(
        output_dir,
        "window_metrics.csv",
    )

    window_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nStatistics successfully calculated.")
    print(f"Processed windows: {len(window_rows)}")
    print(f"CSV generated: {output_path}")

    return window_df