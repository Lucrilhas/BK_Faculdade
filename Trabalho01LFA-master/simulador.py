from automato import Automato

# Classe simulador que ira verifica quais palavras funcionam em um automato
class Simulador:
    def __init__(self, aut: Automato, prints=False):
        self.aut = aut  # Recebe o automato
        self.prints = prints # Prints de ajuda

    def inicia(self):   # Incia o procedimento
        # Verifica cada palavra e salva a resposta em resultados do automato
        for palavra in self.aut.palavras:

            if self.prints:
                print(f"Início da palavra: {palavra}")
            self.aut.resultados.append(self.resolve_palavra(palavra))
            if self.prints:
                print("\nFim da palavra\n")

    # Verificador se a palavra funciona no automato
    def resolve_palavra(self, palavra):
        # Primeiro verifica se a palavra tem transições fora do alfabeto
        for p in palavra:
            if p not in self.aut.vals:
                if self.prints:
                    print("Essa palavra foi recusada por ter valores fora do alfabeto.")
                return "Essa palavra contém valores fora do alfabeto do automato"

        # Inicia uma fila de operações
        fila = [(self.aut.e_inicial[0], palavra)]  # (estado atual, palavra restante)
        if self.prints:
            print(f"Fila inicial com estado incial e Palavra completa: \n{fila}")

        # Enquanto a fila não estiver vazia
        while fila:
            if self.prints:
                print(f"Nova iteração, processamento com a cabeça da lista: {fila[0]}")
            # Caso a primeira operação da fila ainda tenha letras na palavra para analisar
            if fila[0][1]:
                if self.prints:
                    print("Cabeça da Fila ainda tem letras na palavra:")
                for transicao in self.aut.r_transicao[fila[0][0]][fila[0][1][0]]:   # Verifica cada transição da primeira operação da fila
                    # Cria uma nova operação com o proximo estado e com a palavra sem a letra que fez chegar aqui
                    # Coloca essa nova operação na fila
                    if self.prints:
                        print(f"Regra de transição encontrada, adicionando à fila: {(transicao, fila[0][1][1:])}")
                    fila.append((transicao, fila[0][1][1:]))
                if self.prints:
                    print(f"Fila após processamento normal: \n{fila}\n")

            # Caso o Automato seja um E-NFA percorre todos os epsilons sem custar nenhuma letra
            if self.aut.tipo == 'e-nfa':
                if self.prints:
                    print("Automato é uma e-NFA, verificando epsilons no estado atual")
                # Faz basicamente a mesma coisa que acima
                for transicao in self.aut.r_transicao[fila[0][0]]['E']:
                    if self.prints:
                        print(f"Adicionando a fila: {(transicao, fila[0][1][:])}")
                    fila.append((transicao, fila[0][1][:]))
                if self.prints:
                    print(f"Fila após processamento epsilon: \n{fila}\n")


            # Se a operação atual é de um estado final
            # E se não tem mais palavra pra percorrer
            # Então a palavra terminou em um estado final, logo é aceita
            if self.prints:
                print("Cabeça da lista é estado final e sem mais letras para tratar?")
            if fila[0][0] in self.aut.e_final and not fila[0][1]:
                if self.prints:
                    print("Sim! A palavra é aceita")
                return "O automato consegue ler essa palavra"

            if self.prints:
                print(f"Não! Continuando...\nRemove-se a cabeça da fila: {fila[0]}")
            # Remove a operação atual da fila
            fila.pop(0)
            if self.prints:
                print(f"Estado da fila: {fila}")

        # Quando a fila fica vazia vem pra cá e a palavra não é suficiente
        return "O automato NÃO consegue ler essa palavra"
