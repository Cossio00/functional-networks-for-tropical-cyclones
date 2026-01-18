import calculate_mean_climatology as cli
from dictionary import CYCLONES

#   A partir deste código, todo o processo para a geração das métricas e plotagem é rodado. 
#   Basta ajustar a região e o ciclone desejado
REGION = "Bengal_Bay"
CYCLONE = "Gaja"

def main():
    if CYCLONE not in CYCLONES:
        print(f"Ciclone '{CYCLONE}' não encontrado no dicionário de ciclones.")
        print("Verifique o ciclone selecionado e tente novamente.")
        return

    cli.calculate_mean_climatology(REGION)

if __name__ == "__main__":
    main()
