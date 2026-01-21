import xarray as xr
import numpy as np
from scipy.stats import kendalltau
from itertools import combinations
import pickle

# Esse código calcula a correlação Tau de Kendall a partir das janelas deslizantes

def calculate_kendall(region, cyclone):
    # ------------------------------------------------------------------
    # 1. ABRIR AS JANELAS
    # ------------------------------------------------------------------
    antes   = xr.open_dataarray(f"Metrics/{region}/{cyclone}/windows_before_{cyclone}.nc")
    durante = xr.open_dataarray(f"Metrics/{region}/{cyclone}/windows_during_{cyclone}.nc")


    def calcula_kendall_gupta(janela_da):
        data = janela_da.values                    
        ocean_mask = ~np.isnan(data).all(axis=0)   
        series = data[:, ocean_mask]               
        N = series.shape[1]
        print(f"Processando {N} pontos oceânicos...")

        tau_matrix = np.zeros((N, N))
        p_matrix   = np.ones((N, N))

        for i, j in combinations(range(N), 2):
            #ignora NaNs automaticamente
            tau, p = kendalltau(series[:, i], series[:, j])
            tau_matrix[i, j] = tau_matrix[j, i] = tau
            p_matrix[i, j]   = p_matrix[j, i]   = p

        # Significância p < 0.05
        sig = p_matrix < 0.05
        tau_sig = tau_matrix.copy()
        tau_sig[~sig] = 0

        # Top 5% dos τ significativos
        upper_tri = tau_sig[np.triu_indices_from(tau_sig, k=1)]
        valores_sig = upper_tri[upper_tri != 0]
        threshold_95 = np.percentile(valores_sig, 95) if len(valores_sig) > 0 else 0

        # Calculo da matriiz de adjacência
        adj = np.zeros_like(tau_matrix)
        adj[tau_sig >= threshold_95] = 1
        
        print(f"   → τ significativos: {len(valores_sig)}")
        print(f"   → Threshold 95º: {threshold_95:.4f}")
        print(f"   → Arestas na rede: {int(adj.sum() / 2)}")

        # Coordenadas dos pontos oceânicos
        lat_2d = np.repeat(janela_da.latitude.values[:, np.newaxis], 68, axis=1)
        lon_2d = np.repeat(janela_da.longitude.values[np.newaxis, :], 41, axis=0)
        lat_ocean = lat_2d[ocean_mask]
        lon_ocean = lon_2d[ocean_mask]

        return {
            'adjacency_matrix': adj,
            'tau_significativo': tau_sig,
            'threshold_95': threshold_95,
            'ocean_mask': ocean_mask,
            'lat_ocean': lat_ocean,
            'lon_ocean': lon_ocean,
            'N_ocean': N
        }

    print("=== JANELA ANTES ===")
    res_antes = calcula_kendall_gupta(antes)

    print("\n=== JANELA DURANTE ===")
    res_durante = calcula_kendall_gupta(durante)

    # ------------------------------------------------------------------
    # 4. SALVAR
    # ------------------------------------------------------------------
    with open(f"Metrics/{region}/{cyclone}/{cyclone}_metrics.pkl", "wb") as f:
        pickle.dump({'antes': res_antes, 'durante': res_durante}, f)
