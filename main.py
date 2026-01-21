import calculate_mean_climatology as cli
import calculate_anomaly as ano
import apply_land_sea_mask as lsm
import create_sliding_windows as sw
import calculate_kendall as kendall
import calculate_degree as degree
import calculate_mean_geographical_distance as mean_dist
import calculate_clustering_coefficient as clust
import boundary_effects_correction as corr
import plot as plt


from dictionary import CYCLONES

#   A partir deste código, todo o processo para a geração das métricas e plotagem é rodado. 
#   Basta ajustar a região e o ciclone desejados. 
REGION = "Bengal_Bay"
CYCLONE = "Gaja"

def main():
    if CYCLONE not in CYCLONES:
        print(f"Ciclone '{CYCLONE}' não encontrado no dicionário de ciclones.")
        print("Verifique o ciclone selecionado e tente novamente.")
        return

    cli.calculate_mean_climatology(REGION)
    ano.calculate_anomaly(REGION, CYCLONE)
    lsm.apply_land_sea_mask(REGION, CYCLONE)
    sw.create_sliding_windows(REGION, CYCLONE)
    kendall.calculate_kendall(REGION, CYCLONE)
    degree.calculate_degree(REGION, CYCLONE)
    mean_dist.calculate_mean_distance(REGION, CYCLONE)
    clust.calculate_clustering(REGION, CYCLONE)
    corr.boundary_correction(REGION, CYCLONE)
    plt.plot(REGION, CYCLONE)


if __name__ == "__main__":
    main()
