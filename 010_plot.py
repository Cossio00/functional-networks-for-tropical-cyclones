import pickle
import numpy as np
import matplotlib.pyplot as plt


# Falta ajustar o tamanho das figuras, posição dos títulos e subtítulos, correção de valores (após aplicar SERN) e 
# tentar melohrar aplicação da máscara

# ------------------------------------------------------------------
# 1. Carregar pickle
# ------------------------------------------------------------------
with open("Metrics/Gaja/kendall_resultados_gaja.pkl", "rb") as f:
    data = pickle.load(f)

antes   = data['antes']
durante = data['durante']

# ------------------------------------------------------------------
# 2. Colocar valores só no oceano
# ------------------------------------------------------------------
def vector_to_grid(values, ocean_mask):
    grid = np.full(ocean_mask.shape, np.nan)  
    grid[ocean_mask] = values
    return grid

ocean_mask = antes['ocean_mask']  

deg_a   = vector_to_grid(np.log10(antes['degree'] + 1), ocean_mask)
deg_d   = vector_to_grid(np.log10(durante['degree'] + 1), ocean_mask)
dist_a  = vector_to_grid(antes['mean_dist'], ocean_mask)
dist_d  = vector_to_grid(durante['mean_dist'], ocean_mask)
clus_a  = vector_to_grid(antes['clustering'], ocean_mask)
clus_d  = vector_to_grid(durante['clustering'], ocean_mask)

# ------------------------------------------------------------------
# 3. COORDENADAS CORRETAS 
# ------------------------------------------------------------------

lon_centers = np.linspace(49.5, 100.0, 68)                    # 49.5 → 100°E
lat_centers = np.linspace(34.5, 4.5, 41)                      # 34.5 → 4.5°N (ordem decrescente)
Lon_c, Lat_c = np.meshgrid(lon_centers, lat_centers, indexing='xy')

# ------------------------------------------------------------------
# 4. Plot FINAL
# ------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(19, 10.5), constrained_layout=True, 
                         subplot_kw={'projection': None})  # sem projeção especial por enquanto

fig.suptitle("Figura 2 – Gupta et al. (2021) reproduzida\nCiclone Tropical Gaja (2018)",
             fontsize=17, fontweight='bold', y=0.98)

metrics = [
    (deg_a,  deg_d,  "Degree (log$_{10}$(degree + 1))",          0.7,  2.1), #Corrigir essas métricas após a correção de bordas
    (dist_a, dist_d, "Mean geographical distance (km)",         300,  1100),
    (clus_a, clus_d, "Local clustering coefficient",           0.38, 0.96),
]

periods = ["Antes do Gaja\n(29 out – 07 nov 2018)", "Durante o Gaja\n(10–19 nov 2018)"]
letters = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

for col, (f1, f2, title, vmin, vmax) in enumerate(metrics):
    for row, field in enumerate([f1, f2]):
        ax = axes[row, col]
        im = ax.pcolormesh(Lon_c, Lat_c, field,
                           cmap='coolwarm', vmin=vmin, vmax=vmax, shading='auto')
        
        #ax.set_xlim(49.5, 100)      # domínio real
        #ax.set_ylim(4.5, 34.5)      # domínio real     Verificar o motivo das coordenadas se desajustarem aqui
        ax.set_xlabel("Longitude (°E)", fontsize=11)
        if col == 0:
            ax.set_ylabel("Latitude (°N)", fontsize=11)
        
        ax.set_title(f"{letters[row*3 + col]} {title}\n{periods[row]}",
                     fontsize=12, pad=12)
        
        cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
        cbar.set_label(title.split("(")[0].strip(), fontsize=10)

for ax in axes.flat:
    ax.set_facecolor('gray')  # opcional: cor para terra
    

plt.savefig("Plots/FIGURA_GAJA.png", dpi=600, bbox_inches='tight', facecolor='white')
plt.show()