import pickle
import numpy as np
import networkx as nx


REGION = "Bengal_Bay"
CYCLONE = "Luban"
# ------------------------------------------------------------------
# 1. Carregar o pickle com a matriz de adjacência correta
# ------------------------------------------------------------------
with open(f"Metrics/{REGION}/{CYCLONE}/{CYCLONE}_metrics.pkl", "rb") as f:
    data = pickle.load(f)

antes   = data['antes']
durante = data['durante']

# Criar os grafos (não-direcionados, binários)
G_antes   = nx.from_numpy_array(antes['adjacency_matrix'])
G_durante = nx.from_numpy_array(durante['adjacency_matrix'])

# ------------------------------------------------------------------
# 2. Calcular o coeficiente de agrupamento local
# ------------------------------------------------------------------
clustering_antes   = np.array([nx.clustering(G_antes,   i) for i in range(G_antes.number_of_nodes())])
clustering_durante = np.array([nx.clustering(G_durante, i) for i in range(G_durante.number_of_nodes())])

# Nós com degree 0 ou 1 recebem clustering = 0.0 automaticamente (comportamento padrão do NetworkX)

# ------------------------------------------------------------------
# 3. Salvar no pickle
# ------------------------------------------------------------------
antes['clustering']   = clustering_antes
durante['clustering'] = clustering_durante

with open(f"Metrics/{REGION}/{CYCLONE}/{CYCLONE}_metrics.pkl", "wb") as f:
    pickle.dump({'antes': antes, 'durante': durante}, f)

print("LOCAL CLUSTERING COEFFICIENT CALCULADO COM SUCESSO!")
print(f"   • Clustering médio ANTES:   {np.mean(clustering_antes):.3f}")
print(f"   • Clustering médio DURANTE: {np.mean(clustering_durante):.3f}")
print(f"   • Mínimo/Máximo ANTES:      {clustering_antes.min():.3f} / {clustering_antes.max():.3f}")
print(f"   • Mínimo/Máximo DURANTE:    {clustering_durante.min():.3f} / {clustering_durante.max():.3f}")