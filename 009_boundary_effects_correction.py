import pickle
import numpy as np
from tqdm import tqdm

def haversine_vec(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

# ------------------------------------------------------------------
# Carregar pickle original
# ------------------------------------------------------------------
with open("Metrics/Gaja/kendall_resultados_gaja.pkl", "rb") as f:
    data = pickle.load(f)
antes   = data['antes']
durante = data['durante']
lat_ocean = np.array(antes['lat_ocean'])
lon_ocean = np.array(antes['lon_ocean'])
n_nodes = len(lat_ocean)

# ------------------------------------------------------------------
# Matriz de distâncias reais entre todos os pontos oceânicos
# ------------------------------------------------------------------
print("Calculando matriz de distâncias reais (1280x1280)...")
coords = np.column_stack((lat_ocean, lon_ocean))
lat1 = coords[:, 0][:, None]
lon1 = coords[:, 1][:, None]
lat2 = coords[:, 0][None, :]
lon2 = coords[:, 1][None, :]
D_real = haversine_vec(lat1, lon1, lat2, lon2)  # matriz 1280×1280 em km

# ------------------------------------------------------------------
# Função SERN: rede aleatória com mesma distribuição de comprimentos
# ------------------------------------------------------------------
def generate_sern(adj_original):
    link_lengths = D_real[adj_original > 0]
    n_links = len(link_lengths)
    new_adj = np.zeros((n_nodes, n_nodes))
   
    np.random.shuffle(link_lengths)
    placed = 0
    tolerance = 10.0  # km
   
    for length in link_lengths:
        i_candidates, j_candidates = np.where(np.abs(D_real - length) < tolerance)
        valid = [(i, j) for i, j in zip(i_candidates, j_candidates) if i < j and new_adj[i, j] == 0]
        if valid:
            i, j = valid[np.random.randint(len(valid))]
            new_adj[i, j] = new_adj[j, i] = 1
            placed += 1
    return new_adj


def correct_metric(raw_values, adj_matrix, name, n_surrogates=400):
    print(f"Corrigindo {name} ({n_surrogates} SERNs)...")
    surrogate_means = np.zeros(n_nodes)
   
    for _ in tqdm(range(n_surrogates), desc=name, leave=False):
        surr_adj = generate_sern(adj_matrix)
       
        if name == "degree":
            values = surr_adj.sum(axis=1).astype(float) 
           
        elif name == "clustering":
            values = np.zeros(n_nodes)
            degrees = surr_adj.sum(axis=1)
            for i in range(n_nodes):
                if degrees[i] < 2:
                    values[i] = 0.0
                    continue
                neighbors = np.where(surr_adj[i])[0]
                # Links entre vizinhos: submatriz dos vizinhos
                links_between = surr_adj[np.ix_(neighbors, neighbors)].sum()
                values[i] = links_between / (degrees[i] * (degrees[i] - 1))
           
        elif name == "mean_dist":
            values = np.zeros(n_nodes)
            for i in range(n_nodes):
                neigh = np.where(surr_adj[i] > 0)[0]
                if len(neigh) > 0:
                    values[i] = np.mean(D_real[i, neigh])
                else:
                    values[i] = np.nan
       
        surrogate_means += values
   
    surrogate_means /= n_surrogates
    corrected = raw_values / (surrogate_means + 1e-12)
    print(f"{name} corrigido!")
    return corrected

# ------------------------------------------------------------------
# Aplicar correção nas 6 métricas
# ------------------------------------------------------------------
antes['degree_corr']     = correct_metric(antes['degree'],     antes['adjacency_matrix'],     "degree")
antes['clustering_corr'] = correct_metric(antes['clustering'], antes['adjacency_matrix'],     "clustering")
antes['mean_dist_corr']  = correct_metric(antes['mean_dist'],  antes['adjacency_matrix'],     "mean_dist")

durante['degree_corr']     = correct_metric(durante['degree'],     durante['adjacency_matrix'], "degree")
durante['clustering_corr'] = correct_metric(durante['clustering'], durante['adjacency_matrix'], "clustering")
durante['mean_dist_corr']  = correct_metric(durante['mean_dist'],  durante['adjacency_matrix'], "mean_dist")

# Salvar
with open("Metrics/Gaja/gaja_corrigido.pkl", "wb") as f:
    pickle.dump({'antes': antes, 'durante': durante}, f)

print("\nCORREÇÃO CONCLUÍDA E SALVA!")