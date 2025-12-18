import xarray as xr

#Cria as janelas de Gaja apresentadas por Gupta, sendo uma antes de Gaja e uma durante Gaja

print("Abrindo arquivo de anomalias com máscara oceânica...")
anoms = xr.open_dataarray("Metrics/Gaja/mslp_anomalies_oct_nov_2018_ocean.nc")

print(f"Total de timesteps no arquivo: {len(anoms.time)}")

janela_antes   = anoms.sel(time=slice("2018-10-29T00:00", "2018-11-07T21:00"))  # 80 passos
janela_durante = anoms.sel(time=slice("2018-11-10T00:00", "2018-11-19T21:00"))  # 80 passos

print(f"Janela antes:   {len(janela_antes.time)} timesteps")
print(f"Janela durante: {len(janela_durante.time)} timesteps")

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

var_encoding = {anoms.name: {'zlib': True, 'complevel': 5}}

janela_antes.to_netcdf(
    "Metrics/Gaja/anomalias_janela_antes_gaja_2018.nc",
    encoding={**var_encoding, **time_encoding},
    engine='netcdf4'  
)

janela_durante.to_netcdf(
    "Metrics/Gaja/anomalias_janela_durante_gaja_2018.nc",
    encoding={**var_encoding, **time_encoding},
    engine='netcdf4'
)