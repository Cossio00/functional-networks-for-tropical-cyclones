import cdsapi
from dictionary_regions import REGIONS

REGION = "NAO"
region = REGIONS[REGION]

client = cdsapi.Client(
    url="https://cds.climate.copernicus.eu/api",
    key= "***********************************",
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

output = f"Dataset/{region["folder"]}/land_sea/land_sea_mask.nc"
client.retrieve(
    dataset,
    request,
    output
)

