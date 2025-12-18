import pickle
import numpy as np
from math import radians, sin, cos, sqrt, atan2

# ------------------------------------------------------------------
# 1. Carregar o pickle com adjacência + coordenadas (gerado pelo Kendall)
# ------------------------------------------------------------------
with open("Metrics/Gaja/kendall_resultados_gaja.pkl", "rb") as f:
    data = pickle.load(f)

antes   = data['antes']
durante = data['durante']

lat = antes['lat_ocean']      
lon = antes['lon_ocean']

adj_antes   = antes['adjacency_matrix']
adj_durante = durante['adjacency_matrix']

# ------------------------------------------------------------------
# 2. Função Haversine 
# ------------------------------------------------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # raio da Terra em km
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

# ------------------------------------------------------------------
# 3. Calcular distância média para cada nó (só para vizinhos conectados)
# ------------------------------------------------------------------
def calc_mean_distance(adj_matrix, lat, lon):
    N = len(lat)
    mean_dist = np.zeros(N)
    for i in range(N):
        neighbors = np.where(adj_matrix[i] == 1)[0]      # índices dos vizinhos
        if len(neighbors) == 0:
            mean_dist[i] = np.nan                        # nó isolado → NaN
            continue
        dists = [haversine(lat[i], lon[i], lat[j], lon[j]) for j in neighbors]
        mean_dist[i] = np.mean(dists)
    return mean_dist

mean_dist_antes   = calc_mean_distance(adj_antes,   lat, lon)
mean_dist_durante = calc_mean_distance(adj_durante, lat, lon)

# ------------------------------------------------------------------
# 4. Salvar no pickle
# ------------------------------------------------------------------
antes['mean_dist']   = mean_dist_antes
durante['mean_dist'] = mean_dist_durante

with open("Metrics/Gaja/kendall_resultados_gaja.pkl", "wb") as f:
    pickle.dump({'antes': antes, 'durante': durante}, f)

print("DISTÂNCIA GEOGRÁFICA MÉDIA CALCULADA COM SUCESSO!")
print(f"   • Distância média ANTES:   {np.nanmean(mean_dist_antes):.0f} km")
print(f"   • Distância média DURANTE: {np.nanmean(mean_dist_durante):.0f} km")
print(f"   • Mínimo/Máximo ANTES:     {np.nanmin(mean_dist_antes):.0f} / {np.nanmax(mean_dist_antes):.0f} km")
print(f"   • Mínimo/Máximo DURANTE:   {np.nanmin(mean_dist_durante):.0f} / {np.nanmax(mean_dist_durante):.0f} km")
print("   → Arquivo salvo: kendall_resultados_COM_mean_dist_gaja.pkl")