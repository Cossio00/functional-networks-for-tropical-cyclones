import xarray as xr
import numpy as np
import glob
import os
from collections import defaultdict
from datetime import datetime

# Este codigo é responsável por calcular a climatologia média
# A climatologia média pega todos os valores de mslp de 1979 a 2018 e calcula uma média para cada dia do ano

def calculate_mean_climatology(region):
    print("chegou")
    data_dir = f"Dataset/{region}/mslp"
    output_file = f"Metrics/{region}/climatology_1979_2018_daily.nc"
    file_list = sorted(glob.glob(os.path.join(data_dir, "mslp_dias_*.nc")))
    print(f"Encontrados {len(file_list)} arquivos.")

    # ================================
    # PASSO 1: Acumular soma e contagem por (mês, dia)
    # ================================
    sums = defaultdict(lambda: np.zeros((41, 68), dtype=np.float64))
    counts = defaultdict(int)

    # Pegar grade de um arquivo qualquer
    sample_ds = xr.open_dataset(file_list[0])
    lat = sample_ds['latitude'].values
    lon = sample_ds['longitude'].values
    n_lat, n_lon = len(lat), len(lon)
    sample_ds.close()

    # ================================
    # PASSO 2: Processar cada arquivo e separar por dia do calendário
    # ================================
    for file_path in file_list:
        basename = os.path.basename(file_path)
        print(f"Processando {basename}...")
        
        ds = xr.open_dataset(file_path)
        
        # Variável MSLP
        mslp = ds['msl'] if 'msl' in ds else ds['mslp']
        
        # Para cada timestamp
        for t in range(len(ds['time'])):
            time = ds['time'].isel(time=t)
            month = int(time.dt.month.values)
            day = int(time.dt.day.values)
            key = (month, day)
            
            # Acumular
            sums[key] += mslp.isel(time=t).values
            counts[key] += 1
        
        ds.close()

    # ================================
    # PASSO 3: Calcular média por dia do calendário
    # ================================
    climatology = np.full((n_lat, n_lon, 366), np.nan, dtype=np.float32)  # 366 para incluir 29/fev

    for (month, day), total_sum in sums.items():
        try:
            ref_date = datetime(2020 if month != 2 or day != 29 else 2020, month, day)
            doy = ref_date.timetuple().tm_yday
            if doy <= 366:
                climatology[:, :, doy - 1] = (total_sum / counts[(month, day)]).astype(np.float32)
                print(f"{day:02d}/{month:02d} → dia do ano {doy} | {counts[(month, day)]} valores")
            else:
                print(f"Ignorando {day:02d}/{month:02d} → doy {doy} (fora do limite)")
        except ValueError as e:
            print(f"Ignorando {day:02d}/{month:02d} (data inválida: {e})")

    # ================================
    # PASSO 4: Salvar (apenas 365 dias, ignorando 366 se vazio)
    # ================================
    doy_coords = np.arange(1, 367)

    # Criar DataArray com 366, mas salvar só até 365 se 366 estiver vazio
    valid_doy = np.where(~np.isnan(climatology).all(axis=(0,1)))[0] + 1  # doys com dados
    clim_da = xr.DataArray(
        climatology[:, :, :365],  
        coords={'latitude': lat, 'longitude': lon, 'dayofyear': np.arange(1, 366)},
        dims=['latitude', 'longitude', 'dayofyear'],
        name='mslp_climatology',
        attrs={
            'units': 'Pa',
            'long_name': 'Daily MSLP Climatology 1979-2018',
            'method': 'Mean of all 3-hourly values for each calendar day (1979-2018)',
            'note': 'Leap day (Feb 29) averaged into Feb 28 or ignored'
        }
    )

    clim_da.to_netcdf(output_file)
    print(f"\nCLIMATOLOGIA SALVA: {output_file} (365 dias)")