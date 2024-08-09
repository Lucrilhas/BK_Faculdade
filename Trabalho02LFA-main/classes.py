
class Automato:
    def __init__(self):
        self.estados = []
        self.vals_es = []
        self.vals_pi = []
        self.pil_ini = []
        self.e_ini = []
        self.e_fin = []
        self.palavras = []
        self.resultados = []
        self.transicoes = []
        self.list_transicoes = []

    def printa(self):
        print(f"Estados: {self.estados}\n"
              f"Estado inicial: {self.e_ini}\n"
              f"Estados finais: {self.e_fin}\n"
              f"Alfabeto de entrada: {self.vals_es}\n"
              f"Alfabeto de pilha: {self.vals_pi}\n"
              f"Pilha inicial: {self.pil_ini}\n"
              f"Palavras:")
        for p in self.palavras:
            print(p)
        print("Funções de Trasição:")
        for k in self.transicoes:
            print(k, end=' : ')
            for i in self.transicoes[k]:
                print(i, self.transicoes[k][i], end='')
            print()

    def organiza(self):
        self.estados = sorted(self.estados)
        self.e_fin = sorted(self.e_fin)
        self.vals_pi = sorted(self.vals_pi)
        self.vals_in = sorted(self.vals_es)
        self.palavras = sorted(sorted(self.palavras), key=len)


class Pilha:
    def __init__(self, lista):
        self._lista = lista

    def __str__(self):
        return str(self._lista)

    def __repr__(self):
        return str(self._lista)

    def __copy__(self):
        return Pilha(self._lista[:])

    def empilha(self, char):
        if char != 'E':
            self._lista.insert(0, char)

    def desempilha(self):
        return self._lista.pop(0) if len(self._lista) else None

