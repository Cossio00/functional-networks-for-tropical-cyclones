import xarray as xr
import numpy as np
import glob
import os
import pandas as pd
from dictionary import CYCLONES

# ================================
# CONFIGURAÇÃO DINÂMICA
# ================================
REGION = "Bengal_Bay"
CYCLONE = "Luban" 

if CYCLONE not in CYCLONES:
    raise ValueError(f"Ciclone '{CYCLONE}' não encontrado no dicionário CYCLONES.")

# Caminhos
clim_path = f"Metrics/{REGION}/climatology_1979_2018_daily.nc"
mslp_dir = f"Dataset/{REGION}/mslp"
output_dir = f"Metrics/{REGION}/{CYCLONE}"
os.makedirs(output_dir, exist_ok=True)

# Carrega climatologia única da região
clim = xr.open_dataarray(clim_path)

# ================================
# FUNÇÃO PARA CALCULAR ANOMALIA DE UM PERÍODO
# ================================
def calculate_anomalies_for_period(start_date, end_date, period_name):
    print(f"\nCalculando anomalias para '{period_name}': {start_date} a {end_date}")
    
    start_pd = pd.Timestamp(start_date)
    end_pd = pd.Timestamp(end_date)
    
    all_files = sorted(glob.glob(os.path.join(mslp_dir, "mslp_*.nc")))
    if not all_files:
        raise FileNotFoundError(f"Nenhum arquivo MSLP encontrado em {mslp_dir}")
    
    print(f"Arquivos MSLP encontrados: {len(all_files)}")
    
    anomaly_list = []
    time_list = []
    
    for file_path in all_files:
        ds = xr.open_dataset(file_path)
        
        # Filtra apenas o período desejado
        ds_period = ds.sel(time=slice(start_pd, end_pd))
        if len(ds_period.time) == 0:
            ds.close()
            continue
        
        mslp = ds_period['msl'] if 'msl' in ds_period else ds_period['mslp']
        
        for t in range(len(ds_period.time)):
            time_val = ds_period['time'][t].values
            dt = pd.Timestamp(time_val)
            
            # Garante que está dentro do intervalo
            if not (start_pd <= dt <= end_pd):
                continue
                
            doy = dt.dayofyear
            if doy > 365:  # 29/fev → usa dia 365 (28/fev)
                doy = 365
            
            try:
                clim_day = clim.sel(dayofyear=doy)
                anomaly = mslp.isel(time=t) - clim_day
                anomaly_list.append(anomaly.values)
                time_list.append(dt)
            except Exception as e:
                print(f"  Erro em {dt}: {e}")
                continue
        
        ds.close()
    
    if not anomaly_list:
        raise ValueError(f"Nenhum dado encontrado para o período '{period_name}'")
    
    # Verifica número de timesteps
    found_timesteps = len(anomaly_list)
    days_span = (end_pd - start_pd).days + 1
    expected_timesteps = days_span * 8  # 8 horários por dia
    
    # Monta DataArray
    anomalies = np.stack(anomaly_list)
    times = np.array(time_list)
    
    anomaly_da = xr.DataArray(
        anomalies,
        coords={'time': times, 'latitude': clim.latitude, 'longitude': clim.longitude},
        dims=['time', 'latitude', 'longitude'],
        name='mslp_anomaly',
        attrs={
            'units': 'Pa',
            'long_name': f'MSLP Anomaly - {period_name}',
            'cyclone': CYCLONE,
            'period': f"{start_date} to {end_date}",
            'method': 'Daily climatology subtracted',
            'reference': 'Gupta et al. (2021)'
        }
    ).sortby('time')
    
    # Salva arquivo separado
    output_file = os.path.join(output_dir, f"{CYCLONE}_anomalies_{period_name}.nc")
    anomaly_da.to_netcdf(output_file)
    print(f"  Anomalias '{period_name}' salvas em: {output_file}")
    print(f"  Shape: {anomaly_da.shape}")

# ================================
# EXECUTA PARA "ANTES" E "DURANTE"
# ================================
calculate_anomalies_for_period(
    CYCLONES[CYCLONE]["antes"][0],
    CYCLONES[CYCLONE]["antes"][1],
    "before"
)

calculate_anomalies_for_period(
    CYCLONES[CYCLONE]["durante"][0],
    CYCLONES[CYCLONE]["durante"][1],
    "during"
)

print(f"\nCálculo de anomalias concluído para o ciclone {CYCLONE}!")