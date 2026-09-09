# README


1. **_calculate_mean_climatology_** - Calcula a climatologia média de MSLP (ou qualquer dataset)

2. **_calculate_anomaly_** - Calcula a anomalia de MSLP a partir da climatologia média (no caso ali somente para as datas
                            antes e durante Gaja)

3. **_apply_land_sea_mask_** - Aplicação da máscara terra-mar, tornando Nan os valores sobre a terra.

4. **_create_windows_gaja_** - Cria as janelas (antes e depois) de Gaja.

5. **_calculate_kendall_** - Calcula o tau de Kendall e armazena em um arquivo pkl

6. **_calculate_degree_** - Calcula grau dos nós e armazena em um arquivo pkl

7. **_calculate_mean_geographical_distance_** - Calcula distância geográfica média e armazena em um arquivo pkl

8. **_calculate_clustering_coefficient_** - Calcula coeficiente de agrupamento e armazena em um arquivo pkl

9. **_boundary_effects_correction_** - Aplica correção de efeitos de borda às métricas calculadas utilizando SERN

10. **_plot_** - Realiza a plotagem dos resultados.


## Regiões e Ciclones Avaliados:

>Baía de Bengala:
>>+ Ciclone Gaja (10/11/2018 a 19/11/2018)<br>
>>+ Ciclone Luban (06/10/2018 a 15/10/2018)<br>
>>+ Ciclone Titli (08/10/2018 a 12/10/2018)<br>
>>+ Vardah (06/12/2016 a 13/12/2016)<br>
>>+ Megh (05/11/2015 a 10/11/2015)