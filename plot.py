import pickle
import numpy as np
import matplotlib.pyplot as plt
from dictionary_regions import REGIONS


def plot(region, cyclone):
    # ======================================================
    # CARREGA MÉTRICAS
    # ======================================================

    region_info = REGIONS[region]
    lat_max, lon_min, lat_min, lon_max = region_info["area"]

    with open(f"Metrics/{region}/{cyclone}/{cyclone}_metrics.pkl", "rb") as f:
        data = pickle.load(f)

    has_antes = "antes" in data
    nrows = 2 if has_antes else 1

    if has_antes:
        antes = data["antes"]
    durante = data["durante"]

    # ======================================================
    # FUNÇÃO 1D → 2D
    # ======================================================
    def vector_to_grid(values, ocean_mask):
        grid = np.full(ocean_mask.shape, np.nan)
        grid[ocean_mask] = values
        return grid

    ocean_mask = durante["ocean_mask"]

    # ======================================================
    # MÉTRICAS
    # ======================================================
    if has_antes:
        deg_a  = vector_to_grid(antes["degree_corr"], ocean_mask)
        dist_a = vector_to_grid(antes["mean_dist_corr"], ocean_mask)
        clus_a = vector_to_grid(antes["clustering_corr"], ocean_mask)

    deg_d  = vector_to_grid(durante["degree_corr"], ocean_mask)
    dist_d = vector_to_grid(durante["mean_dist_corr"], ocean_mask)
    clus_d = vector_to_grid(durante["clustering_corr"], ocean_mask)

    # ======================================================
    # COORDENADAS
    # ======================================================
    lat = durante["lat"]
    lon = durante["lon"]

    Lon_c, Lat_c = np.meshgrid(lon, lat)


    # ======================================================
    # FIGURA
    # ======================================================
    fig, axes = plt.subplots(
        nrows, 3,
        figsize=(13, 3.2 if nrows == 1 else 6),
        constrained_layout=True
    )

    if nrows == 1:
        axes = axes[np.newaxis, :]

    # ======================================================
    # ESCALAS
    # ======================================================
    def get_vmin_vmax(arr1, arr2, buffer=0.05):
        vals = np.concatenate([arr1.ravel(), arr2.ravel()])
        vals = vals[~np.isnan(vals)]
        vmin, vmax = vals.min(), vals.max()
        return (
            vmin - buffer * (vmax - vmin),
            vmax + buffer * (vmax - vmin)
        )

    if has_antes:
        metrics = [
            ("Degree (corrected)", deg_a, deg_d, *get_vmin_vmax(deg_a, deg_d)),
            ("Mean geographical distance (corrected)", dist_a, dist_d, *get_vmin_vmax(dist_a, dist_d)),
            ("Local clustering coefficient (corrected)", clus_a, clus_d, 0.6, 2.1),
        ]
        letters = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    else:
        metrics = [
            ("Degree (corrected)", deg_d),
            ("Mean geographical distance (corrected)", dist_d),
            #("Local clustering coefficient (corrected)", clus_d, 0.8, 3.2), #Luban
            #("Local clustering coefficient (corrected)", clus_d, 0.9, 2.55), #Vardah
            ("Local clustering coefficient (corrected)", clus_d, 0.9, 2.3), #Megh
            #("Local clustering coefficient (corrected)", clus_d, 1.05, 3.45), #Irma
        ]
        letters = ['(a)', '(b)', '(c)']

    # ======================================================
    # PLOT
    # ======================================================
    cbar_images = {}

    for col, item in enumerate(metrics):

        if has_antes:
            title, f1, f2, vmin, vmax = item
            fields = [f1, f2]
        else:
            if len(item) == 2:
                title, field = item
                vmin, vmax = np.nanmin(field), np.nanmax(field)
            else:
                title, field, vmin, vmax = item
            fields = [field]

        for row, field in enumerate(fields):
            ax = axes[row, col]

            im = ax.pcolormesh(
                Lon_c, Lat_c, field,
                cmap="RdBu_r",
                vmin=vmin,
                vmax=vmax,
                shading="auto"
            )

            ax.set_xlim(lon_min, lon_max)
            ax.set_ylim(lat_min, lat_max)
            ax.set_facecolor("gray")

            ax.text(
                -0.12, 0.95,
                letters[row * 3 + col],
                transform=ax.transAxes,
                fontsize=14,
                fontweight="bold",
                va="top",
                ha="right"
            )

            if row == len(fields) - 1:
                cbar_images[col] = im

    # ======================================================
    # COLORBARS
    # ======================================================
    for col, im in cbar_images.items():
        fig.colorbar(
            im,
            ax=axes[:, col],
            orientation="horizontal",
            fraction=0.046,
            pad=0.08
        )

    # ======================================================
    # SALVAR
    # ======================================================
    plt.savefig(
        f"Plots/{region}/Plot_{cyclone}.png",
        dpi=600,
        bbox_inches="tight",
        facecolor="white"
    )
    plt.show()
