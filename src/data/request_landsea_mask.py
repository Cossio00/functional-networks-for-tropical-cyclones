import cdsapi
import os
from dotenv import load_dotenv

from config import REGIONS


def request_landsea_mask(region_name):
    
    print("\nIniciando obtenção da máscara terra-mar...")

    if region_name not in REGIONS:
        raise ValueError(
            f"Região '{region_name}' não encontrada no dicionário de regiões."
        )

    region = REGIONS[region_name]

    load_dotenv()
    api_key = os.getenv("API_KEY")

    if not api_key:
        raise ValueError(
            "API_KEY não encontrada nas variáveis de ambiente."
        )


    client = cdsapi.Client(
        url="https://cds.climate.copernicus.eu/api",
        key= api_key,
        verify=True,
    )

    dataset = "reanalysis-era5-single-levels"
    request = {
        "product_type": ["reanalysis"],
        "variable": ["land_sea_mask"],
        "year": ["2018"],
        "month": ["11"],
        "day": ["10"],
        "time": ["00:00"],
        "data_format": "netcdf_legacy",
        "download_format": "unarchived",
        "area": region["area"],
        "grid": "0.75/0.75"
        }

    output_dir = f"Dataset/{region["folder"]}/land_sea"
    os.makedirs(output_dir, exist_ok=True)

    output_file = f"{output_dir}/land_sea_mask.nc"

    client.retrieve(
        dataset,
        request,
        output_file
    )

    print("\nMáscara terra-mar obtida com sucesso!")

