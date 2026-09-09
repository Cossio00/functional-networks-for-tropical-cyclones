import xarray as xr
import os
from config import CYCLONES


def create_sliding_windows(region, cyclone):

    print("\nCriando janelas deslizantes...")

    input_file = (f"Metrics/{region}/{cyclone}/{cyclone}_ocean_anomalies.nc")

    if not os.path.exists(input_file):
        raise FileNotFoundError(
            f"Arquivo de anomalias não encontrado:\n{input_file}"
        )

    print(f"Abrindo arquivo de anomalias:\n{input_file}")

    anomalies = xr.open_dataarray(input_file)

    start = CYCLONES[cyclone]["start"]
    end = CYCLONES[cyclone]["end"]

    print(f"\nPeríodo definido no dicionário:\n{start} → {end}")

    anomalies_period = anomalies.sel(
        time=slice(start, end)
    )

    n_timesteps = len(anomalies_period.time)

    if n_timesteps == 0:
        anomalies.close()

        raise ValueError(
            f"Nenhum timestep encontrado entre "
            f"{start} e {end}."
        )

    if n_timesteps > 1:

        time_values = anomalies_period.time.values

        differences = (time_values[1:] - time_values[:-1])

        differences_hours = (differences / 1e9 / 3600)

        if not all(differences_hours == 3):

            anomalies.close()

            raise ValueError(
                "Os dados não estão em resolução de 3 horas.\n"
                f"Intervalos encontrados: {differences_hours}"
            )

    window_days = 10
    timesteps_per_day = 8

    window_size = (
        window_days * timesteps_per_day
    )

    step = timesteps_per_day

    if n_timesteps < window_size:

        anomalies.close()

        raise ValueError(
            f"O intervalo definido possui "
            f"{n_timesteps} timesteps, "
            f"mas são necessários "
            f"{window_size} timesteps "
            f"para uma janela de {window_days} dias."
        )
    
    n_windows = ((n_timesteps - window_size) // step) + 1

    print(
        f"\nConfiguração:"
        f"\n  Intervalo: {start} → {end}"
        f"\n  Timesteps: {n_timesteps}"
        f"\n  Tamanho da janela: {window_days} dias"
        f"\n  Timesteps por janela: {window_size}"
        f"\n  Avanço entre janelas: 1 dia"
        f"\n  Total de janelas: {n_windows}"
    )

    windows_dir = (f"Metrics/{region}/{cyclone}/windows")

    os.makedirs(windows_dir, exist_ok=True)

    old_files = [
        file
        for file in os.listdir(windows_dir)
        if file.startswith("window_")
        and file.endswith(".nc")
    ]

    for file in old_files:

        os.remove(
            os.path.join(windows_dir, file)
        )

    if old_files:

        print(f"\nRemovidas {len(old_files)} janelas antigas.")

    for i in range(n_windows):

        start_idx = i * step
        end_idx = start_idx + window_size

        window = anomalies_period.isel(
            time=slice(
                start_idx,
                end_idx
            )
        )


        if len(window.time) != window_size:

            anomalies.close()

            raise ValueError(
                f"Janela {i + 1} possui "
                f"{len(window.time)} timesteps, "
                f"mas deveria possuir "
                f"{window_size}."
            )

        window_start = window.time.values[0]
        window_end = window.time.values[-1]

        window_file = (f"window_{i + 1:03d}.nc")

        window_path = os.path.join(windows_dir, window_file)

        window.to_netcdf(
            window_path
        )

        print(f"Janela {i + 1:03d}: {window_start} → {window_end} ({len(window.time)} timesteps)"
        )

    anomalies_period.close()
    anomalies.close()

    print(f"\n{n_windows} janela(s) criada(s) com sucesso!")

    print(f"Diretório: {windows_dir}")