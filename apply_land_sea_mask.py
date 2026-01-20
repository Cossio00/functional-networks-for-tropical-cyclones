import xarray as xr
import os

def apply_land_sea_mask(region, cyclone):
    # Caminhos
    anomalies_antes_file = f"Metrics/{region}/{cyclone}/{cyclone}_anomalies_before.nc"
    anomalies_durante_file = f"Metrics/{region}/{cyclone}/{cyclone}_anomalies_during.nc"
    mask_file = f"Dataset/{region}/land_sea/land_sea_mask.nc"

    output_dir = f"Metrics/{region}/{cyclone}"
    os.makedirs(output_dir, exist_ok=True)

    output_antes_ocean = os.path.join(output_dir, f"{cyclone}_ocean_anomalies_before.nc")
    output_durante_ocean = os.path.join(output_dir, f"{cyclone}_ocean_anomalies_during.nc")

    # ================================
    # 1. CARREGAR MÁSCARA TERRA-MAR
    # ================================
    print("Carregando máscara terra-mar...")
    mask_ds = xr.open_dataset(mask_file)

    if 'lsm' in mask_ds:
        mask = mask_ds['lsm']
    elif 'land_sea_mask' in mask_ds:
        mask = mask_ds['land_sea_mask']
    else:
        raise KeyError("Variável de máscara não encontrada. Verifique o nome no arquivo.")

    # Remove dimensão time se existir e squeeze
    if 'time' in mask.dims:
        mask = mask.isel(time=0, drop=True)
    mask = mask.squeeze()

    print(f"Máscara carregada: {mask.shape} (lat, lon)")

    # ================================
    # 2. FUNÇÃO PARA APLICAR MÁSCARA E SALVAR
    # ================================
    def apply_and_save_ocean_mask(input_file, output_file, period_name):
        print(f"\nAplicando máscara ao período '{period_name}'...")
        print(f"Abrindo: {input_file}")
        
        anomaly_da = xr.open_dataarray(input_file)
        
        # Interpola máscara para a grade das anomalias
        mask_interp = mask.interp(latitude=anomaly_da.latitude, longitude=anomaly_da.longitude, method='nearest')
        
        # Máscara binária: True = oceano (lsm < 0.5)
        ocean_mask = mask_interp < 0.5
        
        # Aplica máscara
        anomaly_ocean = anomaly_da.where(ocean_mask)
        
        # Atributos
        anomaly_ocean.attrs.update({
            'cyclone': cyclone,
            'region': region,
            'period': period_name,
            'mask_applied': 'ocean_only (land points set to NaN)',
            'ocean_points': int(ocean_mask.sum().item()),
            'reference': 'Gupta et al. (2021)'
        })
        
        # Encoding para compressão
        encoding = {
            'mslp_anomaly': {'zlib': True, 'complevel': 5},
            'time': {
                'units': 'days since 1970-01-01 00:00:00',
                'calendar': 'gregorian'
            }
        }
        
        # Salva arquivo separado
        anomaly_ocean.to_netcdf(output_file, encoding=encoding)
        print(f"Salvo: {output_file}")
        print(f"   Pontos oceânicos: {int(ocean_mask.sum().item())}")
        print(f"   Timesteps: {len(anomaly_ocean.time)}")
        
        anomaly_da.close()

    # ================================
    # 3. APLICA PARA ANTES E DURANTE
    # ================================
    apply_and_save_ocean_mask(anomalies_antes_file, output_antes_ocean, "antes")
    apply_and_save_ocean_mask(anomalies_durante_file, output_durante_ocean, "durante")

    mask_ds.close()

    print(f"\nMÁSCARA OCEÂNICA APLICADA COM SUCESSO!")
    print(f"Dois arquivos gerados para {cyclone}:")
    print(f"   {output_antes_ocean}")
    print(f"   {output_durante_ocean}")