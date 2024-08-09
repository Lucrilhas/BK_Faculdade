def caminho_minimo(matriz, taxa, origem, destino):
    # inicializacoes
    origem -= 1
    destino -= 1
    cidades = [c for c in range(len(matriz)) if c != origem and c != destino]

    caminhos = [[origem, *c, destino] for c in get_caminhos(cidades)]
    caminhos = valida_caminhos(matriz, caminhos)

    melhor_custo = None
    melhor_caminho = None

    for caminho_atual in caminhos:
        custo_atual = 0
        for indc in range(len(caminho_atual)):
            if indc != len(caminho_atual) - 1:
                custo_atual += matriz[caminho_atual[indc]][caminho_atual[indc + 1]]
                if 0 < indc < len(caminho_atual) - 1:
                    custo_atual += taxa[caminho_atual[indc]]

        if melhor_custo is None or custo_atual < melhor_custo:
            melhor_custo = custo_atual
            melhor_caminho = caminho_atual.copy()

    return [i + 1 for i in melhor_caminho], melhor_custo


def valida_caminhos(matriz, caminhos):
    pos_invalidos = []
    for i in range(len(matriz)):
        for j in range(len(matriz)):
            if matriz[i][j] == -1:
                pos_invalidos.append([i, j])

    resultado = []
    for c in caminhos:
        c_ok = True
        for n in range(len(c) - 1):
            # print([c[n], c[n+1]])
            if [c[n], c[n + 1]] in pos_invalidos:
                c_ok = False
                break
        if c_ok:
            resultado.append(c)

    return resultado


def aumenta_caminho(caminho, opcoes):
    for indc in range(len(caminho) - 1, -1, -1):
        if caminho[indc] == None:
            caminho[indc] = opcoes[0]
            return caminho
        elif caminho[indc] == opcoes[-1]:
            caminho[indc] = opcoes[0]
        else:
            caminho[indc] = opcoes[opcoes.index(caminho[indc]) + 1]
            return caminho


def get_caminhos(opcoes):
    caminho = [None] * len(opcoes)
    caminhos = []
    resultado = [[]]
    while True:
        caminho = aumenta_caminho(caminho, opcoes)
        if caminho is None:
            break
        caminhos.append(caminho.copy())

    for c in caminhos:
        while None in c:
            c.pop(0)
        if not any(c.count(element) > 1 for element in c):
            resultado.append(c)
    return resultado


if __name__ == '__main__':
    c, r = caminho_minimo(
        [[0, 3, 22, -1, 4], [3, 0, 5, -1, -1], [22, 5, 0, 9, 20], [-1, -1, 9, 0, 4], [4, -1, 20, 4, 0]],
        [5, 17, 8, 3, 1],
        2,
        4
    )

    print(c)
    print(r)

    # caminho_minimo(
    #     [[0, 3, 22, -1], [3, 0, 5, -1], [22, 5, 0, 9], [-1, -1, 9, 0]],
    #     [5, 17, 8, 3],
    #     1,
    #     3
    # )

    # op = [1, 2, 3]
    # c = get_caminhos(3, op)
    #
    # for a in c:
    #     print(a)
