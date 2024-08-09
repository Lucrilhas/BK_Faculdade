# Classe Automato, que serve praticamente apenas para organizar os dados pra não ficar jogado em uma lista
class Automato:
    def __init__(self):
        self.estados = []
        self.r_transicao = []
        self.e_inicial = []
        self.e_final = []
        self.vals = []
        self.tipo = 'dfa'
        self.palavras = []
        self.resultados = []

    # Imprime os valores do automato
    def printa_automato(self):
        print(f"Estados: {sorted(self.estados)}\n"
              f"Estado inicial: {self.e_inicial}\n"
              f"Estado final: {self.e_final}\n"
              f"Alfabeto: {self.vals}\n"
              f"Palavras: {self.palavras}\n"
              f"Tipo: {self.tipo}\n"
              f"Resultados: {self.resultados}")
        print("Regras de Transição:")
        for k in self.r_transicao:
            print(k, end=' : ')
            for i in self.r_transicao[k]:
                print(i, self.r_transicao[k][i], end=' ')
            print()

    # Organiza os dados de forma alfabetica
    def organiza(self):
        self.estados = sorted(self.estados)
        self.e_final = sorted(self.e_final)
        self.palavras = sorted(sorted(self.palavras), key=len)
        self.vals = sorted(self.vals)
        for e in self.estados:
            for v in self.vals:
                self.r_transicao[e][v] = sorted(self.r_transicao[e][v])

