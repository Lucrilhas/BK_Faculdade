from CaminhoMinimo import caminho_minimo

def inicio():
    resultado_final = "\n"
    num_paises = int(input())
    input()  # Linha em branco
    for _ in range(num_paises):
        # Variaveis utilizadas
        num_cidades = None
        taxas = None
        matriz = []
        caminhos = []

        # Ler a matriz
        while True:
            matriz.append([int(n) for n in input().split()])
            if matriz[-1][-1] == 0:
                break
        num_cidades = len(matriz)

        # Ler as taxas
        taxas = [int(n) for n in input().split()]

        # Ler os Caminhos
        while True:
            caminhos.append([int(n) for n in input().split()])
            if not caminhos[-1]:
                caminhos.pop(-1)
                break

        # Printa valores
        # print(f"\nValores:\nDataset: {matriz}\nTaxas: {taxas}\nCaminhos: {caminhos}\n")


        # Funcao de calculo de transporte minimo
        for cam in caminhos:
            c, r = caminho_minimo(matriz, taxas, *cam)

            # Salva para sair tudo de uma vez
            resultado_final += f"From {cam[0]} to {cam[1]} :\nPath "
            for indc, val in enumerate(c):
                resultado_final += f"{val}"
                if indc < len(c)-1:
                    resultado_final += "-->"
            resultado_final += f"\nTotal cost : {r}\n\n"



    print(resultado_final)


if __name__ == '__main__':
    inicio()
