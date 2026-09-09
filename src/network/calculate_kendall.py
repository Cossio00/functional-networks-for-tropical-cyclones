import xarray as xr
import numpy as np
from scipy.stats import kendalltau
from itertools import combinations
import pickle
import os


def calculate_kendall(region, cyclone):

    windows_dir = f"Metrics/{region}/{cyclone}/windows"

    if not os.path.exists(windows_dir):
        raise FileNotFoundError(
            f"Diretório de janelas não encontrado: {windows_dir}"
        )

    window_files = sorted(
        [
            file for file in os.listdir(windows_dir)
            if file.endswith(".nc")
        ]
    )

    if not window_files:
        raise FileNotFoundError(
            f"Nenhuma janela encontrada em {windows_dir}"
        )

    print(f"\nEncontradas {len(window_files)} janela(s) para o ciclone {cyclone}.")

    def calcula_kendall(janela):

        data = janela.values

        ocean_mask = ~np.isnan(data).all(axis=0)

        series = data[:, ocean_mask]

        N = series.shape[1]

        print(f"Processando {N} pontos oceânicos...")
        
        tau_matrix = np.zeros((N, N))
        p_matrix = np.ones((N, N))

        for i, j in combinations(range(N), 2):

            tau, p = kendalltau(
                series[:, i],
                series[:, j]
            )

            if np.isnan(tau) or np.isnan(p):
                tau = 0.0
                p = 1.0

            tau_matrix[i, j] = tau
            tau_matrix[j, i] = tau

            p_matrix[i, j] = p
            p_matrix[j, i] = p

        sig = p_matrix < 0.05

        tau_sig = tau_matrix.copy()

        tau_sig[~sig] = 0

        upper_indices = np.triu_indices_from(
            tau_sig,
            k=1
        )

        upper_tri = tau_sig[upper_indices]

        valores_sig = upper_tri[upper_tri > 0]

        if len(valores_sig) > 0:

            threshold_95 = np.percentile(valores_sig, 95)

        else:

            threshold_95 = 0

        adj = np.zeros_like(tau_matrix)

        if len(valores_sig) > 0:

            adj[tau_sig >= threshold_95] = 1

        np.fill_diagonal(adj, 0)

        print(f"   → τ significativos positivos: {len(valores_sig)}")
        print(f"   → Threshold 95º: {threshold_95:.4f}")
        print(f"   → Arestas na rede: {int(adj.sum() / 2)}")

        nlat = len(janela.latitude)
        nlon = len(janela.longitude)

        lat_2d = np.repeat(
            janela.latitude.values[:, np.newaxis],
            nlon,
            axis=1
        )

        lon_2d = np.repeat(
            janela.longitude.values[np.newaxis, :],
            nlat,
            axis=0
        )

        lat_ocean = lat_2d[ocean_mask]
        lon_ocean = lon_2d[ocean_mask]

        return {
            "adjacency_matrix": adj,
            "tau_significativo": tau_sig,
            "tau_matrix": tau_matrix,
            "p_matrix": p_matrix,
            "threshold_95": threshold_95,
            "ocean_mask": ocean_mask,
            "lat_ocean": lat_ocean,
            "lon_ocean": lon_ocean,
            "lat": janela.latitude.values,
            "lon": janela.longitude.values,
            "N_ocean": N
        }

    resultados = {}

    for window_number, window_file in enumerate(window_files, start=1):

        print("\n" + "=" * 70)
        print(f"PROCESSANDO JANELA {window_number:03d}")
        print(f"Arquivo: {window_file}")
        print("=" * 70)

        window_path = os.path.join(windows_dir, window_file)

        janela = xr.open_dataarray(window_path)

        print(f"Período: {janela.time.values[0]} → {janela.time.values[-1]}")
        print(f"Timesteps: {len(janela.time)}")

        resultado = calcula_kendall(janela)

        resultado["window"] = window_number

        resultado["start"] = janela.time.values[0]
        resultado["end"] = janela.time.values[-1]

        resultado["n_timesteps"] = len(janela.time)

        resultados[f"window_{window_number:03d}"] = resultado

        janela.close()

    output_file = (f"Metrics/{region}/{cyclone}/{cyclone}_metrics.pkl")

    with open(output_file,"wb") as f:

        pickle.dump(resultados, f)

    print("\n" + "=" * 70)
    print(f"Cálculo de Kendall concluído para {cyclone}.")
    print(f"Total de redes calculadas: {len(resultados)}")
    print(f"Resultados salvos em: {output_file}")
    print("=" * 70)