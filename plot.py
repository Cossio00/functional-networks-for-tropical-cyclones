import pickle
import numpy as np
import matplotlib.pyplot as plt


def plot(region, cyclone):
    # Carrega o pickle com as métricas corrigidas
    with open(f"Metrics/{region}/{cyclone}/{cyclone}_metrics.pkl", "rb") as f:
        data = pickle.load(f)
    antes = data['antes']
    durante = data['durante']

    # Função para reshape 1D → 2D (só pontos oceânicos)
    def vector_to_grid(values, ocean_mask):
        grid = np.full(ocean_mask.shape, np.nan)
        grid[ocean_mask] = values
        return grid

    ocean_mask = antes['ocean_mask']

    # Métricas corrigidas
    deg_a   = vector_to_grid(antes['degree_corr'], ocean_mask)
    deg_d   = vector_to_grid(durante['degree_corr'], ocean_mask)
    dist_a  = vector_to_grid(antes['mean_dist_corr'], ocean_mask)
    dist_d  = vector_to_grid(durante['mean_dist_corr'], ocean_mask)
    clus_a  = vector_to_grid(antes['clustering_corr'], ocean_mask)
    clus_d  = vector_to_grid(durante['clustering_corr'], ocean_mask)

    # COORDENADAS
    lon_centers = np.linspace(49.5, 100.0, 68)      # 49.5°E → 100°E
    lat_centers = np.linspace(34.5, 4.5, 41)        # 34.5°N → 4.5°N (decrescente como no ERA5)
    Lon_c, Lat_c = np.meshgrid(lon_centers, lat_centers, indexing='xy')


    fig, axes = plt.subplots(2, 3, figsize=(13, 6), constrained_layout=True)

    def get_vmin_vmax(arr1, arr2, buffer=0.05):
        vals = np.concatenate([arr1.ravel(), arr2.ravel()])
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            return 0, 1
        vmin = np.min(vals)
        vmax = np.max(vals)
        return vmin - buffer * (vmax - vmin), vmax + buffer * (vmax - vmin)

    metrics = [
        ("Degree (corrected)", deg_a, deg_d, *get_vmin_vmax(deg_a, deg_d)),
        ("Mean geographical distance (corrected)", dist_a, dist_d, *get_vmin_vmax(dist_a, dist_d)),
        #("Local clustering coefficient (corrected)", clus_a, clus_d, *get_vmin_vmax(clus_a, clus_d)),
        #("Degree (corrected)", deg_a, deg_d, 0.6, 1.8),
        #("Degree (corrected)", deg_a, deg_d, 0.04, 2.72),
        #("Degree (corrected)", deg_a, deg_d, 0.04, 2.45), #Gaja
        #("Degree (corrected)", deg_a, deg_d, 0.0, 4.0), #Luban
        #("Degree (corrected)", deg_a, deg_d, 0.0, 2.5), #Vardah
        #("Mean geographical distance (corrected)", dist_a, dist_d, 0.7, 1.4),
        #("Mean geographical distance (corrected)", dist_a, dist_d, 0.24, 1.83),
        #("Mean geographical distance (corrected)", dist_a, dist_d, 0.24, 1.67), #Gaja
        #("Mean geographical distance (corrected)", dist_a, dist_d, 0.28, 1.72), #Luban
        #("Mean geographical distance (corrected)", dist_a, dist_d, 0.24, 2.4), #Vardah
        #("Local clustering coefficient (corrected)", clus_a, clus_d, 0.8, 2.0),
        #("Local clustering coefficient (corrected)", clus_a, clus_d, 0, 2.75),
        ("Local clustering coefficient (corrected)", clus_a, clus_d, 0.6, 2.1), #Gaja
        #("Local clustering coefficient (corrected)", clus_a, clus_d, 0.8, 3.0) #Luban
        #("Local clustering coefficient (corrected)", clus_a, clus_d, 0.9, 2.45) #Vardah
    ]

    periods = [f"Antes do {cyclone}\n", f"Durante o {cyclone}\n"]
    letters = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

    # Dicionário para guardar as imagens das barras de cor (uma por coluna)
    cbar_images = {}

    for col, (title, f1, f2, vmin, vmax) in enumerate(metrics):
        for row, field in enumerate([f1, f2]):
            ax = axes[row, col]
            im = ax.pcolormesh(Lon_c, Lat_c, field,
                            cmap='RdBu_r', vmin=vmin, vmax=vmax, shading='auto')
            
            ax.set_xlim(49.5, 100.0)
            ax.set_ylim(4.5, 34.5)
            
            ax.text(-0.12, 0.95, letters[row*3 + col], transform=ax.transAxes,
                    fontsize=14, fontweight='bold', va='top', ha='right')
            
            if row == 1: 
                cbar_images[col] = im

    for col, im in cbar_images.items():
        cbar = fig.colorbar(im, ax=axes[:, col], orientation='horizontal',
                            fraction=0.046, pad=0.08, shrink=1)
        cbar.set_label(metrics[col][0].split("(")[0].strip()+" (corrected)", fontsize=10)

    # Terra em cinza
    for ax in axes.flat:
        ax.set_facecolor('gray')

    plt.savefig(f"Plots/{region}/Plot_{cyclone}.png", dpi=600, bbox_inches='tight', facecolor='white')
    plt.show()