import xarray as xr
from dictionary import CYCLONES

def create_sliding_windows(region, cyclone):

    print("Abrindo arquivo de anomalias com máscara oceânica...")
    anoms_before = xr.open_dataarray(f"Metrics/{region}/{cyclone}/{cyclone}_ocean_anomalies_before.nc")
    anoms_during = xr.open_dataarray(f"Metrics/{region}/{cyclone}/{cyclone}_ocean_anomalies_during.nc")


    window_before = anoms_before.sel(time=slice(CYCLONES[cyclone]["antes"][0], CYCLONES[cyclone]["antes"][1]))
    window_during = anoms_during.sel(time=slice(CYCLONES[cyclone]["durante"][0], CYCLONES[cyclone]["durante"][1]))

    print(f"Janela antes:   {len(window_before.time)} timesteps")
    print(f"Janela durante: {len(window_during.time)} timesteps")

    # ===================================================================
    # SALVAR COM TRATAMENTO PARA EVITAR ERRO DE TIME
    # ===================================================================
    time_encoding = {
        'time': {
            'units': 'days since 1970-01-01 00:00:00',
            'calendar': 'gregorian',
            'dtype': 'float64'
        }
    }


    window_before.to_netcdf(
        f"Metrics/{region}/{cyclone}/windows_before_{cyclone}.nc",
        engine='netcdf4'  
    )

    window_during.to_netcdf(
        f"Metrics/{region}/{cyclone}/windows_during_{cyclone}.nc",
        engine='netcdf4'
    )