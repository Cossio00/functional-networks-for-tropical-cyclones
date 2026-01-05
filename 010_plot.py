import pickle
import numpy as np
import matplotlib.pyplot as plt


REGION = "Bengal_Bay"
CYCLONE = "Gaja"
# Carrega o pickle com as métricas corrigidas
with open(f"Metrics/{REGION}/{CYCLONE}/{CYCLONE}_metrics.pkl", "rb") as f:
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

# Plot
fig, axes = plt.subplots(2, 3, figsize=(19, 10.5), constrained_layout=True)
fig.suptitle(f"{CYCLONE}",
             fontsize=17, fontweight='bold', y=0.98)

metrics = [
    #("Degree (corrected)", deg_a, deg_d, 0.6, 1.8),
    #("Degree (corrected)", deg_a, deg_d, 0.04, 2.72),
    ("Degree (corrected)", deg_a, deg_d, 0.04, 2.45),
    #("Mean geographical distance (corrected)", dist_a, dist_d, 0.7, 1.4),
    #("Mean geographical distance (corrected)", dist_a, dist_d, 0.24, 1.83),
    ("Mean geographical distance (corrected)", dist_a, dist_d, 0.24, 1.67),
    #("Local clustering coefficient (corrected)", clus_a, clus_d, 0.8, 2.0),
    #("Local clustering coefficient (corrected)", clus_a, clus_d, 0, 2.75),
    ("Local clustering coefficient (corrected)", clus_a, clus_d, 0.3, 2.1)
]

periods = ["Antes do Gaja\n(29 out – 07 nov 2018)", "Durante o Gaja\n(10–19 nov 2018)"]
letters = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

for col, (title, f1, f2, vmin, vmax) in enumerate(metrics):
    for row, field in enumerate([f1, f2]):
        ax = axes[row, col]
        im = ax.pcolormesh(Lon_c, Lat_c, field,
                           cmap='RdBu_r', vmin=vmin, vmax=vmax, shading='auto')
        
        ax.set_xlim(49.5, 100.0)    # domínio real
        ax.set_ylim(4.5, 34.5)     # domínio real (ajusta visualmente)
        ax.set_xlabel("Longitude (°E)", fontsize=11)
        if col == 0:
            ax.set_ylabel("Latitude (°N)", fontsize=11)
        
        ax.set_title(f"{letters[row*3 + col]} {title}\n{periods[row]}",
                     fontsize=12, pad=15)
        
        cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
        cbar.set_label(title.split("(")[0].strip(), fontsize=10)

# Terra em cinza
for ax in axes.flat:
    ax.set_facecolor('gray')

plt.savefig(f"Plots/{REGION}/Plot_{CYCLONE}.png", dpi=600, bbox_inches='tight', facecolor='white')
plt.show()