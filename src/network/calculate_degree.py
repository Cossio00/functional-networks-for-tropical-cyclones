import pickle
import numpy as np
import networkx as nx


def calculate_degree(region, cyclone):

    metrics_path = (f"Metrics/{region}/{cyclone}/{cyclone}_metrics.pkl")

    with open(metrics_path, "rb") as f:
        data = pickle.load(f)

    print(f"\nCalculando grau para {len(data)} janela(s)...")

    for window_id, window_data in data.items():

        print(f"\nCalculando grau - {window_id}")

        adjacency = window_data["adjacency_matrix"]

        G = nx.from_numpy_array(adjacency)

        degree = np.array([
            G.degree(i)
            for i in range(G.number_of_nodes())
        ])

        window_data["degree"] = degree

        print(f"  Nós: {G.number_of_nodes()}")
        print(f"  Grau médio: {degree.mean():.2f}")

    with open(metrics_path, "wb") as f:
        pickle.dump(data, f)

    print(f"\nGraus calculados e salvos em:")
    print(metrics_path)