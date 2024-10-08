"""O problema da mochila: um problema de otimização combinatória.
O nome dá-se devido ao modelo de uma situação em que é necessário
preencher uma mochila com objetos de diferentes pesos e valores.
O objetivo é que se preencha a mochila com o maior valor possível,
não ultrapassando o peso máximo."""
# RODAR COM PYTHON 3!!!
import random

from genetic2020 import *
import time

# [peso,valor]
t = time.time()
pesos_e_valores = [[4, 30], [8, 10], [8, 30], [25, 75], [2, 10], [50, 100], [6, 300], [12, 50], [100, 400], [8, 300]]
print(pesos_e_valores)
peso_maximo = 200       # Peso maximo da mochila
n_de_cromossomos = 150  # Numero de individuos
geracoes = 10000           # Numero de geracoes
n_de_itens = len(pesos_e_valores)  # Analogo aos pesos e valores

# EXECUCAO DOS PROCEDIMENTOS
populacao = population(n_de_cromossomos, n_de_itens) # Cria uma populaçao inicial
historico_de_fitness = [media_fitness(populacao, peso_maximo, pesos_e_valores)] # Cria um historico dos fitness da populacao para ser plotado

for i in range(geracoes):   # Criam diversas populacoes, evoluindo a cada geracao
    populacao = evolve(populacao, peso_maximo, pesos_e_valores, n_de_cromossomos)
    historico_de_fitness.append(media_fitness(populacao, peso_maximo, pesos_e_valores))

# PRINTS DO TERMINAL\
print(f'Tempo gasto: {time.time() - t}')

# for indice, dados in enumerate(historico_de_fitness):
#     print("Geracao: ", indice, " | Media de valor na mochila: ", dados)
#
# print("\nPeso máximo:", peso_maximo, "g\n\nItens disponíveis:")
# for indice, i in enumerate(pesos_e_valores):
#     print("Item ", indice + 1, ": ", i[0], "g | R$", i[1])
#
# print("\nExemplos de boas solucoes: ")
# for i in range(5):
#     print(populacao[i])

# GERADOR DE GRAFICO
# from matplotlib import pyplot as plt
#
# plt.plot(range(len(historico_de_fitness)), historico_de_fitness)
# plt.grid(True, zorder=0)
# plt.title("Problema da mochila")
# plt.xlabel("Geracao")
# plt.ylabel("Valor medio da mochila")
# plt.savefig('exec_pre10')
# plt.show()