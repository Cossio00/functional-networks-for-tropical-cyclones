import xarray as xr
import numpy as np
import glob
import os
import pandas as pd
from config import CYCLONES


def calculate_anomaly(region, cyclone):

    clim_path = f"Metrics/{region}/climatology_1979_2018_daily.nc"
    mslp_dir = f"Dataset/{region}/mslp"
    output_dir = f"Metrics/{region}/{cyclone}"

    os.makedirs(output_dir, exist_ok=True)

    print("Abrindo climatologia média...")

    clim = xr.open_dataarray(clim_path)
    
    start_date = CYCLONES[cyclone]["start"]
    end_date = CYCLONES[cyclone]["end"]

    print(f"\nCalculando anomalias para '{cyclone}': {start_date} a {end_date}")

    start_pd = pd.Timestamp(start_date)
    end_pd = pd.Timestamp(end_date)

    all_files = sorted(glob.glob(os.path.join(mslp_dir, "mslp_*.nc")))

    if not all_files:
        raise FileNotFoundError(f"Nenhum arquivo MSLP encontrado em {mslp_dir}")

    print(f"Arquivos MSLP encontrados: {len(all_files)}")

    anomaly_list = []
    time_list = []

    for file_path in all_files:

        print(f"Processando: {os.path.basename(file_path)}")

        ds = xr.open_dataset(file_path)

        ds_period = ds.sel(
            time=slice(start_pd, end_pd)
        )

        if len(ds_period.time) == 0:
            ds.close()
            continue

        if "msl" in ds_period:
            mslp = ds_period["msl"]
        elif "mslp" in ds_period:
            mslp = ds_period["mslp"]
        else:
            ds.close()
            raise KeyError(
                f"Variável 'msl' ou 'mslp' não encontrada em "
                f"{file_path}"
            )

        for t in range(len(ds_period.time)):

            time_val = ds_period["time"].isel(time=t).values
            dt = pd.Timestamp(time_val)

            if not (start_pd <= dt <= end_pd):
                continue

            doy = dt.dayofyear

            if doy > 365:
                doy = 365

            try:

                clim_day = clim.sel(
                    dayofyear=doy
                )

                anomaly = (
                    mslp.isel(time=t) - clim_day
                )

                anomaly_list.append(
                    anomaly.values
                )

                time_list.append(dt)

            except Exception as e:

                print(f"Erro ao calcular anomalia em {dt}: {e}")

                continue

        ds.close()

    if not anomaly_list:

        raise ValueError(
            f"Nenhum dado encontrado para o período "
            f"{start_date} a {end_date}"
        )

    anomalies = np.stack(anomaly_list)
    times = np.array(time_list)

    sort_idx = np.argsort(times)

    anomalies = anomalies[sort_idx]
    times = times[sort_idx]

    anomaly_da = xr.DataArray(
        anomalies,
        coords={
            "time": times,
            "latitude": clim.latitude,
            "longitude": clim.longitude
        },
        dims=[
            "time",
            "latitude",
            "longitude"
        ],
        name="mslp_anomaly",
        attrs={
            "units": "Pa",
            "long_name": "MSLP Anomaly",
            "cyclone": cyclone,
            "period": f"{start_date} to {end_date}",
            "method": "Daily climatology subtracted"
        }
    )

    output_file = os.path.join(
        output_dir,
        f"{cyclone}_anomalies.nc"
    )

    anomaly_da.to_netcdf(
        output_file,
        engine="netcdf4"
    )

    print(f"\nAnomalias do ciclone '{cyclone}' salvas em: {output_file}")
    print(f"Período: {start_date} a {end_date}")
    print(f"Timesteps: {len(anomaly_da.time)}")
    print(f"Shape: {anomaly_da.shape}")
    print(f"Resolução temporal: {len(anomaly_da.time)} timestamps")
    print(f"\nCálculo de anomalias concluído para {cyclone}!")