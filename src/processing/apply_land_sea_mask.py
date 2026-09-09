import xarray as xr
import os
import numpy as np

def apply_land_sea_mask(region, cyclone):

    print("\nAplicando máscara terra-mar...")
    
    metrics_dir = f"Metrics/{region}/{cyclone}"
    mask_file = f"Dataset/{region}/land_sea/land_sea_mask.nc"
    
    input_file = os.path.join(metrics_dir, f"{cyclone}_anomalies.nc")
    
    output_file = os.path.join(metrics_dir, f"{cyclone}_ocean_anomalies.nc")
    
    os.makedirs(metrics_dir, exist_ok=True)
    
    if not os.path.exists(input_file):
        print(f"ERRO: Arquivo não encontrado: {input_file}")
        print("Execute o calculate_anomaly.py primeiro.")
        return False
    
    print(f"Carregando máscara: {mask_file}")
    mask_ds = xr.open_dataset(mask_file)
    
    if 'lsm' in mask_ds:
        mask = mask_ds['lsm']
    elif 'land_sea_mask' in mask_ds:
        mask = mask_ds['land_sea_mask']
    else:
        raise KeyError("Variável de máscara não encontrada. Verifique o nome no arquivo.")
    
    if 'time' in mask.dims:
        mask = mask.isel(time=0, drop=True)
    mask = mask.squeeze()
    
    print(f"Máscara carregada: {mask.shape} (lat, lon)")
    
    
    print(f"\nLendo anomalias: {input_file}")
    anomaly_da = xr.open_dataarray(input_file)
    print(f"Anomalias carregadas: {anomaly_da.shape} (time, lat, lon)")
    print(f"Timesteps: {len(anomaly_da.time)}")
    
    print("\nInterpolando máscara...")
    mask_interp = mask.interp(
        latitude=anomaly_da.latitude,
        longitude=anomaly_da.longitude,
        method='nearest'
    )
    
    ocean_mask = mask_interp < 0.5
    
    n_ocean = int(ocean_mask.sum().item())
    n_total = ocean_mask.size
    print(f"Pontos oceânicos: {n_ocean} de {n_total} ({n_ocean/n_total*100:.1f}%)")
    
    if n_ocean == 0:
        print("ERRO: Nenhum ponto oceânico encontrado!")
        anomaly_da.close()
        mask_ds.close()
        return False
    
    print("Aplicando máscara...")
    anomaly_ocean = anomaly_da.where(ocean_mask)
    
    if np.all(np.isnan(anomaly_ocean.values)):
        print("ERRO: Todos os pontos são NaN após aplicar a máscara!")
        anomaly_da.close()
        mask_ds.close()
        return False
    
    anomaly_ocean.attrs.update({
        'cyclone': cyclone,
        'region': region,
        'mask_applied': 'ocean_only (land points set to NaN)',
        'ocean_points': n_ocean,
    })
    
    encoding = {
        'mslp_anomaly': {'zlib': True, 'complevel': 5},
        'time': {
            'units': 'days since 1970-01-01 00:00:00',
            'calendar': 'gregorian'
        }
    }
    
    print(f"\nSalvando: {output_file}")
    anomaly_ocean.to_netcdf(output_file, encoding=encoding)
    
    print(f"  Shape: {anomaly_ocean.shape}")
    print(f"  Timesteps: {len(anomaly_ocean.time)}")
    print(f"  Pontos oceânicos: {n_ocean}")
    
    anomaly_da.close()
    anomaly_ocean.close()
    mask_ds.close()
    
    print("\nMÁSCARA OCEÂNICA APLICADA COM SUCESSO!")
    return True