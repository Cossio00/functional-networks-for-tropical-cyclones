import pickle
import numpy as np
import networkx as nx
from dictionary import CYCLONES


def calculate_degree(region, cyclone):

    print(pickle.__version__)
    # ------------------------------------------------------------------
    # 1. Carregar o pickle gerado anteriormente no cálculo de tau de Kendall
    # ------------------------------------------------------------------
    with open(f"Metrics/{region}/{cyclone}/{cyclone}_metrics.pkl", "rb") as f:
        data = pickle.load(f)
    
    if "antes" in CYCLONES[cyclone]:
        antes   = data['antes']
    durante = data['durante']

    # ------------------------------------------------------------------
    # 2. Criar os grafos a partir das matrizes de adjacência
    # ------------------------------------------------------------------
    if "antes" in CYCLONES[cyclone]:
        G_antes   = nx.from_numpy_array(antes['adjacency_matrix'])
    G_durante = nx.from_numpy_array(durante['adjacency_matrix'])

    # ------------------------------------------------------------------
    # 3. Calcular o degree 
    # ------------------------------------------------------------------
    if "antes" in CYCLONES[cyclone]:
        degree_antes   = np.array([G_antes.degree(i)   for i in range(G_antes.number_of_nodes())])
    degree_durante = np.array([G_durante.degree(i) for i in range(G_durante.number_of_nodes())])

    if "antes" in CYCLONES[cyclone]:
        antes['degree']   = degree_antes
    durante['degree'] = degree_durante

    if "antes" in CYCLONES[cyclone]:
        with open(f"Metrics/{region}/{cyclone}/{cyclone}_metrics.pkl", "wb") as f:
            pickle.dump({'antes': antes, 'durante': durante}, f)
    else:
        with open(f"Metrics/{region}/{cyclone}/{cyclone}_metrics.pkl", "wb") as f:
            pickle.dump({'durante': durante}, f)
    