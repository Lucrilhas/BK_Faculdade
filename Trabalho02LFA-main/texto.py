from classes import Automato


# Funcao que lê a arquivo "entrada.txt" como uma string de forma crua
def le_arq_entrada():
    # Lê o arquivo de entrada:
    with open('entrada.txt') as f:
        conteudo = f.read()

    # Separa as instancias de entrada:
    entradas = []
    aux = ''
    for char in conteudo:  # Adiciona cada caracter a uma string
        if char == '{':  # Delimita cada entrada utilizando Chaves
            aux = ''
        elif char == '}':
            entradas.append(aux)
        else:
            aux += char

    # Chama a função de interpretar a string de cada entrada e a retorna como automato
    return [interpreta_entrada(entrada, Automato()) for entrada in entradas]

# Funcao que interpreta cada automato pegando suas caracteristicas -> Quintupla
def interpreta_entrada(txt_entrada, aut):
    # Interpreta os dados para as variaveis
    get_rt = False
    get_ef = False
    get_pl = False

    for lin in txt_entrada.split('\n'):
        if get_rt:
            if ']' in lin:
                get_rt = False
            else:
                aut.transicoes.append(remove_special_chars(lin).split(','))
        elif r'"transicoes":' in lin:
            get_rt = True

        elif r'"estado_inicial":' in (rms := remove_special_chars(lin)):
            aut.e_ini = [rms.replace(r'"estado_inicial":', '')]

        elif r'"pilha_inicial":' in (rms := remove_special_chars(lin)):
            aut.pil_ini = [rms.replace(r'"pilha_inicial":', '')]

        elif get_ef:
            if ']' in lin:
                get_ef = False
            else:
                aut.e_fin.append(remove_special_chars(lin))
        elif r'"estados_final":' in lin:
            get_ef = True

        elif get_pl:
            if ']' in lin:
                get_pl = False
            else:
                aut.palavras.append(remove_special_chars(lin).split(','))
        elif r'"palavras":' in lin:
            get_pl = True
    for p in aut.palavras:
        p = p[0].split(",")

    for t in aut.transicoes:
        if t[0] not in aut.estados:
            aut.estados.append(t[0])
        if t[1] not in aut.estados:
            aut.estados.append(t[1])
        if t[2] not in aut.vals_es:
            aut.vals_es.append(t[2])
        if t[3] not in aut.vals_pi:
            aut.vals_pi.append(t[3])

    dicionario = {}
    for e in aut.estados:
        dicionario[e] = {}
        for t in aut.transicoes:
            dicionario[e][f"{t[2]}-{t[3]}"] = []

    for ei, ef, pi, pp, c in aut.transicoes:
        dicionario[ei][f"{pi}-{pp}"].append(f"{ef}-{c}")
    aut.list_transicoes = aut.transicoes
    aut.transicoes = dicionario
    aut.organiza()
    return aut

# Retira vários caracteres que nós humanos usamos para organizar mas não são importantes para a máquina
def remove_special_chars(txt):
    return txt.replace(' ', '').replace('\n', '').replace('\t', '')

def automatos_to_text(automatos):
    text = ''

    for aut in automatos:
        text += '{\n'
        text += f'\tEstados: {aut.estados}\n'
        text += f'\tEstado inicial: {aut.e_ini}\n'
        text += f'\tEstado final: {aut.e_fin}\n'
        text += f'\tAlfabeto de entrada: {aut.vals_es}\n'
        text += f'\tAlfabeto de pilha: {aut.vals_pi}\n'
        text += f'\tPilha inicial: {aut.pil_ini}\n'
        text += '\tRegras de transição:[\n\t\t\t\t'
        for ei, ef, ve, vp, c in aut.list_transicoes:
            text += f'{ve}-{vp}\t\t\t\t\t'
        text += '\n'

        for e in aut.estados:
            text += f'\t\t{e}\t\t'
            for ei, ef, ve, vp, c in aut.list_transicoes:
                text += f"{aut.transicoes[e][f'{ve}-{vp}']}\t\t\t\t"
            text += '\n'
        text += '\t]\n'

        text += '\tPalavras:[\n'
        for p, r in zip(aut.palavras, aut.resultados):
            text += f'\t\t{p}\t->\t{r}\n'
        text += '\t]\n'
        text += '}\n\n'

    with open('saida.txt', 'w') as f:
        f.write(text)
