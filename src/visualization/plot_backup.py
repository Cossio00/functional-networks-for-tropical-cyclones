import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

from config import REGIONS
from cyclones_tracks import CYCLONES_TRACKS


def plot(region, cyclone):

    # ======================================================
    # CARREGA INFORMAÇÕES DA REGIÃO
    # ======================================================

    region_info = REGIONS[region]

    lat_max, lon_min, lat_min, lon_max = region_info["area"]

    metrics_path = (
        f"Metrics/{region}/{cyclone}/"
        f"{cyclone}_metrics.pkl"
    )

    with open(metrics_path, "rb") as f:
        data = pickle.load(f)

    if not data:
        raise ValueError(
            f"Nenhuma janela encontrada para {cyclone}."
        )

    print(
        f"\nEncontradas {len(data)} janela(s) "
        f"para o ciclone {cyclone}."
    )

    # ======================================================
    # DEFINE ESCALAS GLOBAIS PARA O CICLONE
    # ======================================================
    #
    # As escalas são calculadas considerando TODAS as
    # janelas do ciclone.
    #
    # Assim, todas as imagens do mesmo ciclone utilizam
    # exatamente a mesma escala para cada métrica.
    #
    # ======================================================

    max_degree = 0
    max_mean_dist = 0
    max_clustering = 0

    for window in data.values():

        degree_values = np.asarray(
            window["degree_corr"]
        )

        mean_dist_values = np.asarray(
            window["mean_dist_corr"]
        )

        clustering_values = np.asarray(
            window["clustering_corr"]
        )

        if np.any(np.isfinite(degree_values)):
            max_degree = max(
                max_degree,
                np.nanmax(degree_values)
            )

        if np.any(np.isfinite(mean_dist_values)):
            max_mean_dist = max(
                max_mean_dist,
                np.nanmax(mean_dist_values)
            )

        if np.any(np.isfinite(clustering_values)):
            max_clustering = max(
                max_clustering,
                np.nanmax(clustering_values)
            )

    # ------------------------------------------------------
    # Adiciona uma pequena margem ao maior valor
    # ------------------------------------------------------

    def add_buffer(value, buffer=0.05):

        if value <= 0:
            return 1

        return value * (1 + buffer)

    max_degree = add_buffer(max_degree)
    max_mean_dist = add_buffer(max_mean_dist)
    max_clustering = add_buffer(max_clustering)

    scales = [
        (0, max_degree),
        (0, max_mean_dist),
        (0, max_clustering)
    ]

    print("\nEscalas utilizadas para o ciclone:")

    print(
        f"  Degree: "
        f"0 → {max_degree:.2f}"
    )

    print(
        f"  Mean geographical distance: "
        f"0 → {max_mean_dist:.2f}"
    )

    print(
        f"  Local clustering coefficient: "
        f"0 → {max_clustering:.2f}"
    )

    # ======================================================
    # CARREGA TRAJETÓRIA DO CICLONE
    # ======================================================

    track_data = CYCLONES_TRACKS.get(cyclone, [])

    track_lats = None
    track_lons = None
    sizes = None

    if track_data:

        track_lats, track_lons, intensities = zip(*track_data)

        max_int = max(intensities)

        sizes = [
            (i / max_int * 50) ** 1.2
            for i in intensities
        ]

    # ======================================================
    # FUNÇÃO PARA TRANSFORMAR VETOR 1D EM GRADE 2D
    # ======================================================

    def vector_to_grid(values, ocean_mask):

        grid = np.full(
            ocean_mask.shape,
            np.nan
        )

        grid[ocean_mask] = values

        return grid

    # ======================================================
    # FUNÇÃO PARA FORMATAR DATA
    # ======================================================

    def format_date(date):

        date = np.datetime64(date)

        date_str = str(date)[:10]

        return (
            f"{date_str[8:10]}/"
            f"{date_str[5:7]}/"
            f"{date_str[0:4]}"
        )

    # ======================================================
    # PASTA DE SAÍDA
    # ======================================================

    output_dir = (
        f"Plots/{region}/{cyclone}"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # ======================================================
    # PROCESSA CADA JANELA
    # ======================================================

    for window_id, window in data.items():

        print(
            f"\nGerando plot para {window_id}..."
        )

        # --------------------------------------------------
        # Número da janela
        # --------------------------------------------------

        window_number = window["window"]

        # --------------------------------------------------
        # Período da janela
        # --------------------------------------------------

        start = format_date(window["start"])
        end = format_date(window["end"])

        period = f"{start} a {end}"

        # --------------------------------------------------
        # Coordenadas
        # --------------------------------------------------

        ocean_mask = window["ocean_mask"]

        lat = window["lat"]
        lon = window["lon"]

        Lon_c, Lat_c = np.meshgrid(
            lon,
            lat
        )

        # --------------------------------------------------
        # Métricas
        # --------------------------------------------------

        deg = vector_to_grid(
            window["degree_corr"],
            ocean_mask
        )

        dist = vector_to_grid(
            window["mean_dist_corr"],
            ocean_mask
        )

        clus = vector_to_grid(
            window["clustering_corr"],
            ocean_mask
        )

        metrics = [
            (
                "Degree (corrected)",
                deg
            ),
            (
                "Mean geographical distance (corrected)",
                dist
            ),
            (
                "Local clustering coefficient (corrected)",
                clus
            )
        ]

        # ==================================================
        # FIGURA
        # ==================================================

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(13, 4),
            constrained_layout=True
        )

        fig.suptitle(
            f"Janela {window_number} — {period}",
            fontsize=16,
            fontweight="bold"
        )

        # --------------------------------------------------
        # PLOTS
        # --------------------------------------------------

        letters = [
            "(a)",
            "(b)",
            "(c)"
        ]

        for col, (
            (title, field),
            (vmin, vmax)
        ) in enumerate(
            zip(metrics, scales)
        ):

            ax = axes[col]

            im = ax.pcolormesh(
                Lon_c,
                Lat_c,
                field,
                cmap="RdBu_r",
                vmin=vmin,
                vmax=vmax,
                shading="auto"
            )

            ax.set_xlim(
                lon_min,
                lon_max
            )

            ax.set_ylim(
                lat_min,
                lat_max
            )

            ax.set_facecolor("gray")

            ax.set_title(
                title,
                fontsize=11
            )

            ax.text(
                -0.12,
                0.95,
                letters[col],
                transform=ax.transAxes,
                fontsize=14,
                fontweight="bold",
                va="top",
                ha="right"
            )

            # --------------------------------------------------
            # TRAJETÓRIA
            # --------------------------------------------------

            if (
                track_lons is not None
                and track_lats is not None
                and sizes is not None
            ):

                ax.scatter(
                    track_lons,
                    track_lats,
                    s=sizes,
                    color="black",
                    alpha=0.8,
                    edgecolor="none"
                )

            # --------------------------------------------------
            # COLORBAR
            # --------------------------------------------------

            fig.colorbar(
                im,
                ax=ax,
                orientation="horizontal",
                fraction=0.046,
                pad=0.08
            )

        # ==================================================
        # SALVAR
        # ==================================================

        filename = (
            f"janela_{window_number:03d}.png"
        )

        output_path = os.path.join(
            output_dir,
            filename
        )

        plt.savefig(
            output_path,
            dpi=600,
            bbox_inches="tight",
            facecolor="white"
        )

        plt.close(fig)

        print(
            f"Plot salvo: {output_path}"
        )

    # ======================================================
    # FINAL
    # ======================================================

    print(
        f"\nTodos os plots do ciclone {cyclone} "
        f"foram gerados com sucesso!"
    )