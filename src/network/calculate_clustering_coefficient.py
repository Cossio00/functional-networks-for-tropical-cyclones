import pickle
import numpy as np
import networkx as nx


def calculate_clustering(region, cyclone):

    metrics_path = f"Metrics/{region}/{cyclone}/{cyclone}_metrics.pkl"

    # ------------------------------------------------------------------
    # 1. Carregar os resultados das redes
    # ------------------------------------------------------------------
    with open(metrics_path, "rb") as f:
        data = pickle.load(f)

    # ------------------------------------------------------------------
    # 2. Calcular o coeficiente de agrupamento para todas as janelas
    # ------------------------------------------------------------------

    for window_id, window_data in data.items():

        print(
            f"Calculando coeficiente de agrupamento - "
            f"janela {window_id}"
        )

        adjacency = window_data["adjacency_matrix"]

        # Criar grafo não direcionado e não ponderado
        G = nx.from_numpy_array(adjacency)

        # Calcular coeficiente de agrupamento local para cada nó
        clustering = np.array([
            nx.clustering(G, i)
            for i in range(G.number_of_nodes())
        ])

        # Adicionar resultado à janela
        window_data["clustering"] = clustering

        print(
            f"  Nós: {G.number_of_nodes()} | "
            f"clustering médio: {clustering.mean():.4f}"
        )

    # ------------------------------------------------------------------
    # 3. Salvar novamente o pickle
    # ------------------------------------------------------------------

    with open(metrics_path, "wb") as f:
        pickle.dump(data, f)

    print(
        f"\nCoeficientes de agrupamento calculados e salvos em: "
        f"{metrics_path}"
    )