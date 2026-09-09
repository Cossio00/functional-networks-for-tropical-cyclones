import cdsapi
import os
from dotenv import load_dotenv

from config import REGIONS


def request_mslp(region_name):

    print("\nIniciando obtenção dos dados de MSLP...")

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
        key=api_key,
        verify=True,
    )

    dataset = "reanalysis-era5-single-levels"
    days = [str(d).zfill(2) for d in range(1, 32)]

    output_dir = f"Dataset/{region['folder']}/mslp"
    os.makedirs(output_dir, exist_ok=True)

    for day in days:
        request = {
            "product_type": ["reanalysis"],
            "variable": ["mean_sea_level_pressure"],
            "year": [str(y) for y in range(1979, 2019)],
            "month": [str(m) for m in range(1, 13) if not (m == 2 and day == "29")],
            "day": [day],
            "time": [
                "00:00", "03:00", "06:00", "09:00",
                "12:00", "15:00", "18:00", "21:00"
            ],
            "data_format": "netcdf_legacy",
            "download_format": "unarchived",
            "area": region["area"],
            "grid": "0.75/0.75"
        }

        output_dir = f"../../Dataset/{region["folder"]}/mslp/mslp_dia_{day}.nc"
        client.retrieve(dataset,request,output_dir)

    
    print("\nObtenção de MSLP concluída!")