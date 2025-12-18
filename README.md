# Code guide

Abaixo está um resumo do que faz cada código:
1. **_calculate_mean_climatology** - Calcula a climatologia média de MSLP (ou qualquer dataset)

2. **_calculate_anomaly_** - Calcula a anomalia de MSLP a partir da climatologia média (no caso ali somente para as datas
                            antes e durante Gaja)

3. **_apply_land_sea_mask_** - Aplicação da máscara terra-mar, tornando Nan os valores sobre a terra.

4. **_create_windows_gaja_** - Cria as janelas (antes e depois) de Gaja.

5. **_calculate_kendall_** - Calcula o tau de Kendall e armazena em um arquivo pkl

6. **_calculate_degree_** - Calcula grau dos nós e armazena em um arquivo pkl

7. **_calculate_mean_geographical_distance_** - Calcula distância geográfica média e armazena em um arquivo pkl

8. **_calculate_clustering_coefficient_** - Calcula coeficiente de agrupamento e armazena em um arquivo pkl

9. **_plot_** - Realiza a plotagem dos resultados.

