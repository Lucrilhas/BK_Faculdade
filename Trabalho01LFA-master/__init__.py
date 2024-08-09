# Autor: Lucas Elias de Andrade Cruvinel
# Codigo para simular DFA, NFA e E-NFA para a disciplina de Linguagens Formais e Automatos
# Com o prof. Dr. Sergio Francisco da Silva

# https://github.com/Lucrilhas/Trabalho01LFA

from textos import *
from simulador import Simulador

# Inicio do codigo
if __name__ == '__main__':
    automatos = le_arq_entrada()    # Le as entradas do arquivo "entrada.txt"

    # Para cada automato realiza uma simulacao
    for n, automato in enumerate(automatos):
        Simulador(automato, prints=False).inicia()    # Passa o automato para a simulacao e inicia

    print('Imprimindo resultados em saida.txt')
    automatos_to_text(automatos)        # Salva os resultados obtidos de todas as entradas no arquivo "saida.txt"
