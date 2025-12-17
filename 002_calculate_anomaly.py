import xarray as xr
import numpy as np
import glob
import os
from datetime import datetime
import pandas as pd

# Esse código é responsável pelo cálculo da anomalia da climatologia
# Em resumo: a anomalia de um momento em um ponto é o valor de mslp naquele momento
# menos a climatologia média daquele ponto naquele dia do ano.

# ================================
# CONFIGURAÇÕES
# ================================
clim_path = "Metrics/Gaja/mslp_climatology_1979_2018_daily.nc"
data_dir = "Dataset/Gaja/mslp"  # Pasta com arquivos MSLP 2018 (ou todos, mas filtraremos 2018)
output_anomaly = "Metrics/Gaja/mslp_anomalies_oct_nov_2018.nc"

# Carregar climatologia
clim = xr.open_dataarray(clim_path)
print(f"Climatologia: {clim.shape} → (lat, lon, dayofyear=1-365)")

# ================================
# 1. Listar TODOS os arquivos MSLP (assumindo múltiplos para out/nov)
# ================================
all_files = sorted(glob.glob(os.path.join(data_dir, "mslp_dias_*.nc")))
print(f"Total arquivos encontrados: {len(all_files)}")

# ================================
# 2. Filtrar e processar APENAS 2018, out/nov
# ================================
anomaly_list = []
time_list = []

for file_path in all_files:
    print(f"Verificando: {os.path.basename(file_path)}")
    ds = xr.open_dataset(file_path)
    
    # Filtrar APENAS 2018
    ds_2018 = ds.sel(time=slice('2018-10-01', '2018-11-30'))  # Só 2018, out/nov
    if len(ds_2018.time) == 0:
        ds.close()
        continue  # Pula arquivos sem 2018
    
    mslp = ds_2018['msl'] if 'msl' in ds_2018 else ds_2018['mslp']
    
    # Para cada timestamp em 2018 out/nov
    for t in range(len(ds_2018['time'])):
        time = ds_2018['time'].isel(time=t)
        month = int(time.dt.month.values)
        day = int(time.dt.day.values)
        
        if month not in [10, 11]:  # Só out/nov
            continue
        
        try:
            doy = datetime(2020, month, day).timetuple().tm_yday
            if doy > 365:
                doy = 365
            
            clim_day = clim.sel(dayofyear=doy)
            anomaly = mslp.isel(time=t) - clim_day
            
            anomaly_list.append(anomaly.values)
            time_list.append(pd.Timestamp(time.values))
            
        except Exception as e:
            print(f"  Erro em {time.values}: {e}")
            continue
    
    ds.close()

# ================================
# 3. Montar DataArray
# ================================
anomalies = np.stack(anomaly_list)
times = np.array(time_list)

anomaly_da = xr.DataArray(
    anomalies,
    coords={'time': times, 'latitude': clim.latitude, 'longitude': clim.longitude},
    dims=['time', 'latitude', 'longitude'],
    name='mslp_anomaly',
    attrs={
        'units': 'Pa',
        'long_name': 'MSLP Anomaly 2018 (out/nov) - Climatology 1979-2018',
        'method': 'Daily climatology subtracted from each 3-hourly value',
        'reference': 'Gupta et al. (2021), Climate Dynamics',
        'period': '2018-10-01 to 2018-11-30'
    }
)

# === ORDENAR TEMPO ===
anomaly_da = anomaly_da.sortby('time')

# ================================
# 4. Salvar + VALIDAÇÃO
# ================================
anomaly_da.to_netcdf(output_anomaly, encoding={'mslp_anomaly': {'zlib': True}})
print(f"\nANOMALIAS SALVAS: {output_anomaly}")
print(f"  Shape: {anomaly_da.shape}")
print(f"  Período: {anomaly_da.time.min().values} → {anomaly_da.time.max().values}")
