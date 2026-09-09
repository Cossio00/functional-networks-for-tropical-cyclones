import pickle
import numpy as np
from math import radians, sin, cos, sqrt, atan2


def calculate_mean_distance(region, cyclone):

    metrics_path = (
        f"Metrics/{region}/{cyclone}/"
        f"{cyclone}_metrics.pkl"
    )

    # ================================================================
    # 1. CARREGAR RESULTADOS
    # ================================================================

    with open(metrics_path, "rb") as f:
        data = pickle.load(f)

    print(
        f"\nCalculando distância geográfica média "
        f"para {len(data)} janela(s)..."
    )

    # ================================================================
    # 2. FUNÇÃO HAVERSINE
    # ================================================================

    def haversine(lat1, lon1, lat2, lon2):

        R = 6371.0  # raio da Terra em km

        lat1, lon1, lat2, lon2 = map(
            radians,
            [lat1, lon1, lat2, lon2]
        )

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = (
            sin(dlat / 2) ** 2
            + cos(lat1)
            * cos(lat2)
            * sin(dlon / 2) ** 2
        )

        c = 2 * atan2(
            sqrt(a),
            sqrt(1 - a)
        )

        return R * c

    # ================================================================
    # 3. PROCESSAR TODAS AS JANELAS
    # ================================================================

    for window_id, window_data in data.items():

        print(
            f"\nCalculando distância geográfica média - "
            f"{window_id}"
        )

        # ------------------------------------------------------------
        # Dados da janela
        # ------------------------------------------------------------

        adjacency = window_data["adjacency_matrix"]

        lat = window_data["lat_ocean"]
        lon = window_data["lon_ocean"]

        N = len(lat)

        # ------------------------------------------------------------
        # Inicializa vetor
        # ------------------------------------------------------------

        mean_dist = np.zeros(N)

        # ------------------------------------------------------------
        # Calcula distância média de cada nó
        # ------------------------------------------------------------

        for i in range(N):

            # Índices dos nós conectados ao nó i
            neighbors = np.where(
                adjacency[i] == 1
            )[0]

            # Nó sem conexões
            if len(neighbors) == 0:

                mean_dist[i] = np.nan

                continue

            # Distâncias até os vizinhos
            distances = [
                haversine(
                    lat[i],
                    lon[i],
                    lat[j],
                    lon[j]
                )
                for j in neighbors
            ]

            # Distância média
            mean_dist[i] = np.mean(distances)

        # ------------------------------------------------------------
        # Adiciona resultado à janela
        # ------------------------------------------------------------

        window_data["mean_dist"] = mean_dist

        # ------------------------------------------------------------
        # Informações
        # ------------------------------------------------------------

        print(
            f"  Nós: {N}"
        )

        print(
            f"  Nós conectados: "
            f"{np.sum(~np.isnan(mean_dist))}"
        )

        print(
            f"  Distância média: "
            f"{np.nanmean(mean_dist):.2f} km"
        )

    # ================================================================
    # 4. SALVAR
    # ================================================================

    with open(metrics_path, "wb") as f:
        pickle.dump(data, f)

    print(
        "\nDistâncias geográficas médias "
        "calculadas e salvas em:"
    )

    print(metrics_path)