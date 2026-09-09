import pickle
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

from . import sern as sn


def boundary_correction(region, cyclone):

    # ================================================================
    # 1. CARREGAR OS RESULTADOS DAS REDES
    # ================================================================

    input_path = (
        f"Metrics/{region}/{cyclone}/"
        f"{cyclone}_metrics.pkl"
    )

    with open(input_path, "rb") as f:
        data = pickle.load(f)

    if not data:
        raise ValueError(
            f"Nenhuma janela encontrada para o ciclone {cyclone}."
        )

    print(
        f"\nEncontradas {len(data)} janela(s) "
        f"para o ciclone {cyclone}."
    )

    # ================================================================
    # 2. FUNÇÃO HAVERSINE
    # ================================================================

    def haversine_vec(lat1, lon1, lat2, lon2):

        from numpy import (
            radians,
            sin,
            cos,
            sqrt,
            arctan2
        )

        R = 6371.0

        lat1, lon1, lat2, lon2 = map(
            radians,
            [lat1, lon1, lat2, lon2]
        )

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = (
            sin(dlat / 2) ** 2
            + cos(lat1)
            * cos(lat2)
            * sin(dlon / 2) ** 2
        )

        c = 2 * arctan2(
            sqrt(a),
            sqrt(1 - a)
        )

        return R * c

    # ================================================================
    # 3. CORRIGIR UMA JANELA
    # ================================================================

    def correct_window(window_id, window):

        print("\n" + "=" * 70)
        print(
            f"PROCESSANDO CORREÇÃO DE BORDA - {window_id}"
        )
        print("=" * 70)

        # ------------------------------------------------------------
        # Coordenadas dos nós
        # ------------------------------------------------------------

        lat_ocean = window["lat_ocean"]
        lon_ocean = window["lon_ocean"]

        n_nodes = len(lat_ocean)

        print(
            f"Número de nós oceânicos: {n_nodes}"
        )

        # ------------------------------------------------------------
        # Matriz de distâncias geográficas
        # ------------------------------------------------------------

        lat1 = lat_ocean[:, None]
        lon1 = lon_ocean[:, None]

        lat2 = lat_ocean[None, :]
        lon2 = lon_ocean[None, :]

        D_real = haversine_vec(
            lat1,
            lon1,
            lat2,
            lon2
        )

        # ============================================================
        # FUNÇÃO PARA GERAR MÉTRICAS DAS SERNs
        # ============================================================

        def compute_surrogate_metrics(
            adj_matrix,
            metric_name,
            n_surrogates=1000
        ):

            # --------------------------------------------------------
            # Distâncias discretizadas
            # --------------------------------------------------------

            D_bin_flat, x = sn.IntegerDistances(
                lat_ocean,
                lon_ocean,
                scale=50.0
            )

            # --------------------------------------------------------
            # Probabilidade de ligação
            # --------------------------------------------------------

            upper_indices = np.triu_indices(
                n_nodes,
                k=1
            )

            A_flat = (
                adj_matrix[upper_indices]
                .astype(int)
            )

            p_bins = sn.LinkProbability(
                A_flat,
                D_bin_flat
            )

            # ========================================================
            # UMA REDE SUBSTITUTA
            # ========================================================

            def single_surrogate(_):

                edges = sn.SernEdges(
                    D_bin_flat,
                    p_bins,
                    n_nodes
                )

                surr_adj = np.zeros(
                    (n_nodes, n_nodes),
                    dtype=int
                )

                surr_adj[
                    edges[:, 0],
                    edges[:, 1]
                ] = 1

                surr_adj[
                    edges[:, 1],
                    edges[:, 0]
                ] = 1

                # --------------------------------------------------
                # DEGREE
                # --------------------------------------------------

                if metric_name == "degree":

                    return np.sum(
                        surr_adj,
                        axis=1
                    ).astype(float)

                # --------------------------------------------------
                # CLUSTERING
                # --------------------------------------------------

                elif metric_name == "clustering":

                    values = np.zeros(
                        n_nodes
                    )

                    degrees = np.sum(
                        surr_adj,
                        axis=1
                    )

                    for i in range(n_nodes):

                        if degrees[i] < 2:

                            values[i] = 0.0
                            continue

                        neighbors = np.nonzero(
                            surr_adj[i]
                        )[0]

                        sub_adj = surr_adj[
                            np.ix_(
                                neighbors,
                                neighbors
                            )
                        ]

                        links_between = (
                            np.sum(sub_adj) // 2
                        )

                        values[i] = (
                            2 * links_between
                        ) / (
                            degrees[i]
                            * (degrees[i] - 1)
                        )

                    return values

                # --------------------------------------------------
                # DISTÂNCIA GEOGRÁFICA MÉDIA
                # --------------------------------------------------

                elif metric_name == "mean_dist":

                    values = np.zeros(
                        n_nodes
                    )

                    for i in range(n_nodes):

                        neighbors = np.nonzero(
                            surr_adj[i]
                        )[0]

                        if len(neighbors) > 0:

                            values[i] = np.mean(
                                D_real[
                                    i,
                                    neighbors
                                ]
                            )

                        else:

                            values[i] = np.nan

                    return values

                else:

                    raise ValueError(
                        f"Métrica desconhecida: "
                        f"{metric_name}"
                    )

            # ========================================================
            # GERAR AS 1000 SERNs
            # ========================================================

            print(
                f"   Corrigindo {metric_name} "
                f"com {n_surrogates} SERNs..."
            )

            results = Parallel(
                n_jobs=-1
            )(
                delayed(single_surrogate)(i)
                for i in tqdm(
                    range(n_surrogates),
                    desc=f"SERN - {metric_name}"
                )
            )

            # ========================================================
            # CONVERTER RESULTADOS
            # ========================================================

            results_array = np.full(
                (
                    n_surrogates,
                    n_nodes
                ),
                np.nan
            )

            for i, result in enumerate(results):

                results_array[i] = result

            # Média das redes substitutas
            surrogate_means = np.nanmean(
                results_array,
                axis=0
            )

            return surrogate_means

        # ============================================================
        # CORREÇÃO DO DEGREE
        # ============================================================

        mean_degree = compute_surrogate_metrics(
            window["adjacency_matrix"],
            "degree",
            n_surrogates=1000
        )

        window["degree_corr"] = (
            window["degree"]
            / (mean_degree + 1e-12)
        )

        # ============================================================
        # CORREÇÃO DO CLUSTERING
        # ============================================================

        mean_clustering = compute_surrogate_metrics(
            window["adjacency_matrix"],
            "clustering",
            n_surrogates=1000
        )

        window["clustering_corr"] = (
            window["clustering"]
            / (mean_clustering + 1e-12)
        )

        # ============================================================
        # CORREÇÃO DA DISTÂNCIA MÉDIA
        # ============================================================

        mean_distance = compute_surrogate_metrics(
            window["adjacency_matrix"],
            "mean_dist",
            n_surrogates=1000
        )

        window["mean_dist_corr"] = (
            window["mean_dist"]
            / (mean_distance + 1e-12)
        )

        print(
            f"\nCorreção concluída para {window_id}."
        )

        return window

    # ================================================================
    # 4. PROCESSAR TODAS AS JANELAS
    # ================================================================

    for window_id in sorted(data.keys()):

        data[window_id] = correct_window(
            window_id,
            data[window_id]
        )

    # ================================================================
    # 5. SALVAR
    # ================================================================

    with open(input_path, "wb") as f:

        pickle.dump(
            data,
            f
        )

    print("\n" + "=" * 70)
    print(
        f"Correção de efeitos de borda concluída para "
        f"{cyclone}."
    )
    print(
        f"Total de janelas corrigidas: "
        f"{len(data)}"
    )
    print(
        f"Resultados salvos em: {input_path}"
    )
    print("=" * 70)