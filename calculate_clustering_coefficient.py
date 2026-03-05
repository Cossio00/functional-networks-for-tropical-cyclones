import pickle
import numpy as np
import networkx as nx
from dictionary import CYCLONES


def calculate_clustering(region, cyclone):
    # ------------------------------------------------------------------
    # 1. Carregar o pickle com a matriz de adjacência
    # ------------------------------------------------------------------
    with open(f"Metrics/{region}/{cyclone}/{cyclone}_metrics.pkl", "rb") as f:
        data = pickle.load(f)

    if "antes" in CYCLONES[cyclone]:
        antes   = data['antes']   
    durante = data['durante']

    # Criar os grafos (não-direcionados, binários)
    if "antes" in CYCLONES[cyclone]:
        G_antes   = nx.from_numpy_array(antes['adjacency_matrix'])
    G_durante = nx.from_numpy_array(durante['adjacency_matrix'])

    # ------------------------------------------------------------------
    # 2. Calcular o coeficiente de agrupamento local
    # ------------------------------------------------------------------
    if "antes" in CYCLONES[cyclone]:
        clustering_antes   = np.array([nx.clustering(G_antes,   i) for i in range(G_antes.number_of_nodes())])
    clustering_durante = np.array([nx.clustering(G_durante, i) for i in range(G_durante.number_of_nodes())])

    # Nós com degree 0 ou 1 recebem clustering = 0.0 automaticamente (comportamento padrão do NetworkX)

    # ------------------------------------------------------------------
    # 3. Salvar no pickle
    # ------------------------------------------------------------------
    if "antes" in CYCLONES[cyclone]:
        antes['clustering']   = clustering_antes
    durante['clustering'] = clustering_durante

    if "antes" in CYCLONES[cyclone]:    
        with open(f"Metrics/{region}/{cyclone}/{cyclone}_metrics.pkl", "wb") as f:
            pickle.dump({'antes': antes, 'durante': durante}, f)
    else:
        with open(f"Metrics/{region}/{cyclone}/{cyclone}_metrics.pkl", "wb") as f:
            pickle.dump({'durante': durante}, f)
