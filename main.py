import sys
import argparse
from src.data import (request_mslp, request_landsea_mask)
from src.processing import (calculate_mean_climatology, calculate_anomaly, apply_land_sea_mask, create_sliding_windows)
from src.network import (calculate_kendall, calculate_degree, calculate_mean_distance, calculate_clustering)
from src.correction import boundary_correction
from src.visualization import plot

from config import CYCLONES, REGIONS

def processing(cyclone):
    if cyclone not in CYCLONES:
        print(f"Ciclone '{cyclone}' não encontrado no dicionário de ciclones.")
        print("Verifique o ciclone selecionado e tente novamente.")
        return
    
    REGION = CYCLONES[cyclone]["region"]
    
    if REGION not in REGIONS:
        print(f"Região '{REGION}' não encontrada no dicionário de regiões.")
        print(f"Verifique se a região do ciclone '{cyclone}' está presente no dicionário de regiões e tente novamente.")
        return   
    
    calculate_mean_climatology(REGION)
    calculate_anomaly(REGION, cyclone)
    apply_land_sea_mask(REGION, cyclone)
    create_sliding_windows(REGION, cyclone)
    calculate_kendall(REGION, cyclone)
    calculate_degree(REGION, cyclone)
    calculate_mean_distance(REGION, cyclone)
    calculate_clustering(REGION, cyclone)
    boundary_correction(REGION, cyclone)
    plot(REGION, cyclone)


def list_cyclones():

    print("\nCiclones disponíveis:\n")

    for cyclone, data in CYCLONES.items():

        region = data["region"]
        start = data["start"]
        end = data["end"]

        print(f"  {cyclone}")
        print(f"    Região:  {region}")
        print(f"    Período: {start} → {end}")
        print()


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Pipeline ERA5 e redes funcionais "
            "para ciclones tropicais."
        )
    )

    parser.add_argument(
        "command",
        nargs="?",
        choices=[
            "processing",
            "mslp",
            "landsea",
            "cyclones"
        ],
        help="Operação a ser executada."
    )

    parser.add_argument(
        "cyclone",
        nargs="?",
        help="Nome do ciclone."
    )

    args = parser.parse_args()

    # ========================================================
    # Nenhum comando
    # ========================================================

    if args.command is None:

        parser.print_help()

        print("\nExemplos:")
        print("  python main.py cyclones")
        print("  python main.py processing CYCLONE_NAME")
        print("  python main.py mslp CYCLONE_NAME")
        print("  python main.py landsea CYCLONE_NAME")

        return

    # ========================================================
    # LISTAR CICLONES
    # ========================================================

    if args.command == "cyclones":

        list_cyclones()

        return

    # ========================================================
    # COMANDOS QUE EXIGEM CICLONE
    # ========================================================

    if args.cyclone is None:

        print(f"\nO comando '{args.command}' requer o nome de um ciclone.")
        print("\nExemplo:")
        print(f"  python main.py {args.command} Gaja")
        print("\nUse:")
        print("  python main.py cyclones")
        print("para verificar os ciclones disponíveis.")

        return

    # ========================================================
    # VERIFICAR CICLONE
    # ========================================================

    if args.cyclone not in CYCLONES:

        print(f"\nCiclone '{args.cyclone}' não encontrado.")
        print("\nCiclones disponíveis:")

        for cyclone in CYCLONES:
            print(f"  - {cyclone}")

        return

    region = CYCLONES[args.cyclone]["region"]

    # ========================================================
    # PROCESSING
    # ========================================================

    if args.command == "processing":

        processing(args.cyclone)

    elif args.command == "mslp":

        request_mslp(region)


    elif args.command == "landsea":

        request_landsea_mask(region)


if __name__ == "__main__":
    main()

