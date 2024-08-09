from classes import Automato, Pilha

class Simulador:
    def __init__(self, aut: Automato, prints=False):
        self.aut = aut
        self.prints = prints

    def inicia(self):
        for palavra in self.aut.palavras:
            self.aut.resultados.append(self.resolve_palavra(palavra))

    def resolve_palavra(self, palavra):
        # print(palavra)
        fila = [(self.aut.e_ini[0], palavra, Pilha(self.aut.pil_ini[:]))]
        # print(fila)
        # print()
        while fila:
            cabeca_pilha = fila[0][2].desempilha()

            if fila[0][1]:
                id_dict = f"{fila[0][1][0]}-{cabeca_pilha}"
                if id_dict in self.aut.transicoes[fila[0][0]].keys():
                    for transicao in self.aut.transicoes[fila[0][0]][id_dict]:
                        novo_estado, comando_pilha = transicao.split("-")
                        nova_pilha = fila[0][2].__copy__()
                        for c in comando_pilha[len(comando_pilha)::-1]:
                            nova_pilha.empilha(c)

                        fila.append((novo_estado, fila[0][1][1:], nova_pilha))

            # print(cabeca_pilha)
            # print(self.aut.transicoes[fila[0][0]].keys())
            if (e_id := f"E-{cabeca_pilha}") in self.aut.transicoes[fila[0][0]].keys():
                for transicao in self.aut.transicoes[fila[0][0]][e_id]:
                    novo_estado, comando_pilha = transicao.split("-")
                    nova_pilha = fila[0][2].__copy__()
                    for c in comando_pilha[len(comando_pilha)::-1]:
                        nova_pilha.empilha(c)
                    fila.append((novo_estado, fila[0][1][:], nova_pilha))

            # print(fila)
            if fila[0][0] in self.aut.e_fin and (len(fila[0][1]) == 0 or fila[0][1][0] == '') and cabeca_pilha == self.aut.pil_ini[0]:
                # print("Final " + str(fila))
                return "O automato consegue ler essa palavra"

            fila.pop(0)
            # print(fila)
            # print()


        return "O automato NÃO consegue ler essa palavra"