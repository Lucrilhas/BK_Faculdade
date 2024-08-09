from texto import le_arq_entrada, automatos_to_text
from simulador import Simulador

if __name__ == '__main__':
    automatos = le_arq_entrada()

    for automato in automatos:
        # automato.printa()
        Simulador(automato, prints=False).inicia()
        # for r in automato.resultados:
        #     print(r)

    print('Imprimindo resultados em saida.txt')
    automatos_to_text(automatos)        # Salva os resultados obtidos de todas as entradas no arquivo "saida.txt"