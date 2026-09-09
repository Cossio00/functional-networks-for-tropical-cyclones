import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

from config import REGIONS
from cyclones_tracks import CYCLONES_TRACKS


# ============================================================
# CONFIGURAÇÃO DOS PERCENTIS
# ============================================================

LOWER_PERCENTILE = 0.5
UPPER_PERCENTILE = 99.5


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
    # CALCULA ESCALAS GLOBAIS DO CICLONE
    # ======================================================
    #
    # A escala é calculada usando os valores de TODAS as
    # janelas do ciclone.
    #
    # Exemplo:
    #
    #   Janela 1 ─┐
    #   Janela 2 ─┤
    #   Janela 3 ─┼──> P2 / P98
    #   Janela 4 ─┤
    #   Janela 5 ─┘
    #
    # Isso garante que todas as janelas possam ser comparadas
    # visualmente.
    # ======================================================

    def get_percentile_scale(metric_name):

        all_values = []

        for window in data.values():

            if metric_name not in window:
                continue

            values = np.asarray(
                window[metric_name],
                dtype=float
            )

            # Remove NaN e infinitos
            values = values[
                np.isfinite(values)
            ]

            if len(values) > 0:
                all_values.append(values)

        if not all_values:
            return 0, 1

        # Junta os valores de todas as janelas
        all_values = np.concatenate(all_values)

        # Calcula os percentis
        vmin = np.percentile(
            all_values,
            LOWER_PERCENTILE
        )

        vmax = np.percentile(
            all_values,
            UPPER_PERCENTILE
        )

        # Evita escala degenerada
        if vmin == vmax:

            return (
                vmin - 1,
                vmax + 1
            )

        return vmin, vmax

    # ======================================================
    # ESCALA DE CADA MÉTRICA
    # ======================================================

    degree_scale = get_percentile_scale(
        "degree_corr"
    )

    mean_dist_scale = get_percentile_scale(
        "mean_dist_corr"
    )

    clustering_scale = get_percentile_scale(
        "clustering_corr"
    )

    scales = [
        degree_scale,
        mean_dist_scale,
        clustering_scale
    ]

    # ======================================================
    # MOSTRA AS ESCALAS CALCULADAS
    # ======================================================

    print("\n" + "=" * 60)
    print("ESCALAS DAS MÉTRICAS")
    print("=" * 60)

    print(
        f"Degree: "
        f"{degree_scale[0]:.4f} "
        f"→ "
        f"{degree_scale[1]:.4f}"
    )

    print(
        f"Mean geographical distance: "
        f"{mean_dist_scale[0]:.4f} "
        f"→ "
        f"{mean_dist_scale[1]:.4f}"
    )

    print(
        f"Clustering coefficient: "
        f"{clustering_scale[0]:.4f} "
        f"→ "
        f"{clustering_scale[1]:.4f}"
    )

    print(
        f"\nPercentis utilizados: "
        f"P{LOWER_PERCENTILE} "
        f"e "
        f"P{UPPER_PERCENTILE}"
    )

    print("=" * 60)

    # ======================================================
    # CARREGA TRAJETÓRIA DO CICLONE
    # ======================================================

    track_data = CYCLONES_TRACKS.get(
        cyclone,
        []
    )

    track_lats = None
    track_lons = None
    sizes = None

    if track_data:

        track_lats, track_lons, intensities = zip(
            *track_data
        )

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

        start = format_date(
            window["start"]
        )

        end = format_date(
            window["end"]
        )

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
        # MÉTRICAS
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

        # ==================================================
        # PLOTS
        # ==================================================

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

            # --------------------------------------------------
            # MAPA
            # --------------------------------------------------

            im = ax.pcolormesh(
                Lon_c,
                Lat_c,
                field,
                cmap="RdBu_r",
                vmin=vmin,
                vmax=vmax,
                shading="auto"
            )

            # --------------------------------------------------
            # LIMITES ESPACIAIS
            # --------------------------------------------------

            ax.set_xlim(
                lon_min,
                lon_max
            )

            ax.set_ylim(
                lat_min,
                lat_max
            )

            # --------------------------------------------------
            # FUNDO
            # --------------------------------------------------

            ax.set_facecolor(
                "gray"
            )

            # --------------------------------------------------
            # TÍTULO
            # --------------------------------------------------

            ax.set_title(
                title,
                fontsize=11
            )

            # --------------------------------------------------
            # LETRA
            # --------------------------------------------------

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

            # ==================================================
            # TRAJETÓRIA DO CICLONE
            # ==================================================

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

            # ==================================================
            # COLORBAR
            # ==================================================

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