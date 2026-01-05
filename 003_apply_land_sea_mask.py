import xarray as xr

# Esse código aplica a máscara terra mar ao mapa com mslp, tornando Nan os valores em porção de terra.

REGION = "Bengal_Bay"
CYCLONE = "Gaja"
# ================================
# CONFIGURAÇÕES
# ================================
anomaly_file = f"Metrics/{REGION}/{CYCLONE}/{CYCLONE}_mslp_anomalies_oct_nov_2018.nc"
mask_file = f"Dataset/{REGION}/land_sea/land_sea_mask.nc"
output_file = f"Metrics/{REGION}/{CYCLONE}/{CYCLONE}_mslp_anomalies_oct_nov_2018_ocean.nc"

# ================================
# 1. CARREGAR ANOMALIAS
# ================================
anoms = xr.open_dataarray(anomaly_file)

# ================================
# 2. CARREGAR E LIMPAR MÁSCARA (remover time:1)
# ================================
print("Carregando máscara terra-mar...")
mask_ds = xr.open_dataset(mask_file)               # Abre como Dataset

# Seleciona a variável (ajuste o nome se for diferente de 'lsm')
mask = mask_ds['lsm']  # ou mask_ds['land_sea_mask'] se for outro nome

# REMOVE DIMENSÃO TIME (e qualquer outra de tamanho 1)
if 'time' in mask.dims:
    mask = mask.isel(time=0, drop=True)             # Remove 'time' completamente
mask = mask.squeeze()                               # Remove todas as dims de tamanho 1 restantes

# ================================
# 3. ALINHAR COORDENADAS
# ================================
mask = mask.interp(latitude=anoms.latitude, longitude=anoms.longitude, method='nearest')

# ================================
# 4. CRIAR MÁSCARA BINÁRIA (True = oceano)
# ================================
ocean_mask = (mask < 0.5)  # Agora garantidamente 2D (lat, lon)

# ================================
# 5. APLICAR MÁSCARA
# ================================
anoms_ocean = anoms.where(ocean_mask)  # Aplica a mesma máscara fixa em todos os 488 timesteps

# ================================
# 6. SALVAR
# ================================
anoms_ocean.attrs.update({
    'mask_applied': 'ocean_only (fixed 2D mask)',
    'land_points': 'set_to_NaN',
    'reference': 'Gupta et al. (2021) - only ocean points used in network',
    'ocean_points': int(ocean_mask.sum().item()),
    'note': 'Land-sea mask squeezed to 2D before applying'
})

# Renomear variável para clareza
anoms_ocean.name = 'mslp_anomaly'

anoms_ocean.to_netcdf(
    output_file
)

print(f"\nMÁSCARA OCEÂNICA APLICADA COM SUCESSO!")
print(f"  Arquivo salvo: {output_file}")
print(f"  Pontos oceânicos mantidos: {int(ocean_mask.sum().item())}")