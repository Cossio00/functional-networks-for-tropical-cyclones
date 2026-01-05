import pickle
import numpy as np
import os
import sys
from tqdm import tqdm
from joblib import Parallel, delayed

import sern as sn #Código obtido do repositório de Rheinwalt para calcular as SERN 


REGION = "Bengal_Bay"
CYCLONE = "Gaja"

# Carrega o pickle original
with open(f"Metrics/{REGION}/{CYCLONE}/{CYCLONE}_metrics.pkl", "rb") as f:
    data = pickle.load(f)

antes = data['antes']
durante = data['durante']

lat_ocean = antes['lat_ocean']
lon_ocean = antes['lon_ocean']
n_nodes = len(lat_ocean)

# Pré-computa matriz de distâncias Haversine (para mean_dist)
def haversine_vec(lat1, lon1, lat2, lon2):
    from numpy import radians, sin, cos, sqrt, arctan2
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * arctan2(sqrt(a), sqrt(1 - a))
    return R * c

lat1 = lat_ocean[:, None]
lon1 = lon_ocean[:, None]
lat2 = lat_ocean[None, :]
lon2 = lon_ocean[None, :]
D_real = haversine_vec(lat1, lon1, lat2, lon2)  

def compute_surrogate_metrics(adj_matrix, name, n_surrogates=1000):
    # Pré-computa bins e probabilidades (uma vez por rede)
    D_bin_flat, x = sn.IntegerDistances(lat_ocean, lon_ocean, scale=50.0)
    upper_indices = np.triu_indices(n_nodes, k=1)
    A_flat = adj_matrix[upper_indices].astype(int)  # Garante 0/1
    p_bins = sn.LinkProbability(A_flat, D_bin_flat)

    def single_surrogate(_):
        edges = sn.SernEdges(D_bin_flat, p_bins, n_nodes)
        surr_adj = np.zeros((n_nodes, n_nodes), dtype=int)
        surr_adj[edges[:, 0], edges[:, 1]] = 1
        surr_adj[edges[:, 1], edges[:, 0]] = 1

        if name == "degree":
            return np.sum(surr_adj, axis=1).astype(float)

        elif name == "clustering":
            values = np.zeros(n_nodes)
            degrees = np.sum(surr_adj, axis=1)
            for i in range(n_nodes):
                if degrees[i] < 2:
                    values[i] = 0.0
                    continue
                neighbors = np.nonzero(surr_adj[i])[0]
                sub_adj = surr_adj[np.ix_(neighbors, neighbors)]
                links_between = np.sum(sub_adj) // 2  # Conta cada aresta uma vez
                values[i] = (2 * links_between) / (degrees[i] * (degrees[i] - 1))
            return values

        elif name == "mean_dist":
            values = np.zeros(n_nodes)
            for i in range(n_nodes):
                neigh = np.nonzero(surr_adj[i])[0]
                if len(neigh) > 0:
                    values[i] = np.mean(D_real[i, neigh])
                else:
                    values[i] = np.nan
            return values

    print(f"Corrigindo {name} com {n_surrogates} SERNs...")
    results = Parallel(n_jobs=-1)(
        delayed(single_surrogate)(i) for i in tqdm(range(n_surrogates))
    )

    # Correção robusta do warning "mean of empty slice"
    results_array = np.full((n_surrogates, n_nodes), np.nan)
    for i, res in enumerate(results):
        results_array[i] = res
    surrogate_means = np.nanmean(results_array, axis=0)

    return surrogate_means

# === Correção para a janela ANTES ===
print("\n=== PROCESSANDO JANELA 'ANTES' ===")
mean_degree_antes = compute_surrogate_metrics(antes['adjacency_matrix'], "degree", n_surrogates=1000)
antes['degree_corr'] = antes['degree'] / (mean_degree_antes + 1e-12)

mean_clust_antes = compute_surrogate_metrics(antes['adjacency_matrix'], "clustering", n_surrogates=1000)
antes['clustering_corr'] = antes['clustering'] / (mean_clust_antes + 1e-12)

mean_dist_antes = compute_surrogate_metrics(antes['adjacency_matrix'], "mean_dist", n_surrogates=1000)
antes['mean_dist_corr'] = antes['mean_dist'] / (mean_dist_antes + 1e-12)

# === Correção para a janela DURANTE ===
print("\n=== PROCESSANDO JANELA 'DURANTE' ===")
mean_degree_durante = compute_surrogate_metrics(durante['adjacency_matrix'], "degree", n_surrogates=1000)
durante['degree_corr'] = durante['degree'] / (mean_degree_durante + 1e-12)

mean_clust_durante = compute_surrogate_metrics(durante['adjacency_matrix'], "clustering", n_surrogates=1000)
durante['clustering_corr'] = durante['clustering'] / (mean_clust_durante + 1e-12)

mean_dist_durante = compute_surrogate_metrics(durante['adjacency_matrix'], "mean_dist", n_surrogates=1000)
durante['mean_dist_corr'] = durante['mean_dist'] / (mean_dist_durante + 1e-12)


output_path = f"Metrics/{REGION}/{CYCLONE}/{CYCLONE}_metrics.pkl"
with open(output_path, "wb") as f:
    pickle.dump({'antes': antes, 'durante': durante}, f)

print(f"\nCorreção de boundary effects concluída!")
print(f"Resultados salvos em: {output_path}")
print("   → Métricas corrigidas: degree_corr, clustering_corr, mean_dist_corr")