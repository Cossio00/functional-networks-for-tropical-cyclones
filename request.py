import cdsapi
from dictionary_regions import REGIONS

REGION = "North_Atlantic_Ocean"

if REGION not in REGIONS:
    print(f"Região '{REGION}' não encontrada no dicionário de regiões.")
    print("Verifique a região selecionada e tente novamente.")
    exit()

region = REGIONS[REGION]

client = cdsapi.Client(
    url="https://cds.climate.copernicus.eu/api",
    key="***********************************",
    verify=True,
)

dataset = "reanalysis-era5-single-levels"
days = [str(d).zfill(2) for d in range(1, 32)]

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

    output = f"Dataset/{region["folder"]}/mslp/mslp_dia_{day}.nc"
    client.retrieve(dataset,request,output)

