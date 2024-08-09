# Simulated Annealing aplicada no problema do caixeiro viajante
# Aluno: Lucas Elias de Andrade Cruvinel
# Estudante da Universidade Federal de Catalão
# lucascruvinel@discente.ufcat.edu.br
# Esse código foi inspirado no código do site https://www.cnblogs.com/youcans/p/14728931.html

import math           
import random      
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

np.set_printoptions(precision=4)
pd.set_option('display.max_rows', 20)
pd.set_option('expand_frame_repr', False)
pd.options.display.float_format = '{:,.2f}'.format

# Função para criar matriz de adjacencia entre as cidades com suas distancias
def getDistMat(nCities, coordinates):
    distMat = np.zeros((nCities,nCities))   # Inicializa uma matriz vazia
    for i in range(nCities):
        for j in range(i,nCities):
            distMat[i][j] = distMat[j][i] = round(np.linalg.norm(coordinates[i]-coordinates[j]))
    return distMat  

# Calcula a distancia da conexao atual
def calTourMileage(tourGiven, nCities, distMat):
    mileageTour = distMat[tourGiven[nCities-1], tourGiven[0]]   
    for i in range(nCities-1):                                  
        mileageTour += distMat[tourGiven[i], tourGiven[i+1]]
    return round(mileageTour)   # Arredonda o valor

# Pertuba o caminho atual trocando dois caminhos entre cidades
def mutateSwap(tourGiven, nCities):
    i = np.random.randint(nCities)          # Cria um valor aleatorio para inicio caminho
    while True:
        j = np.random.randint(nCities)      # Cria um valor aleatorio para destino caminho
        if i!=j: break                      # Forçar destino diferente de inicio 

    # Modificca o caminho dado para o novo modificado e retorna ele
    tourSwap = tourGiven.copy()       
    tourSwap[i],tourSwap[j] = tourGiven[j],tourGiven[i]
    return tourSwap

# Inicio do programa
def main(n):
    # Inicaliza uma série de coordenadas aleatorias para representar a localização das cidades
    coordinates = np.random.randint(1000000, size=(n, 2))


    # Inicializa algumas variaveis do funcionamento do simulated annealing
    tInitial = 100.0        # Temperatura Inicial
    tFinal  = 1             # Temperatura Final
    nMarkov = 1000          # Número de cadeias de Markov, ou iterações
    alfa    = 0.98          # Constante de diminuição de temperatura

    # Variaveis auxiliares
    nCities = coordinates.shape[0]              # Número de cidades
    distMat = getDistMat(nCities, coordinates)  # Cria uma matriz de distancias
    nMarkov = nCities                           # Valor inicial do Markov
    tNow    = tInitial                          # Inicializa var de temperatura 

    # Inicializar listas auxiliares
    tourNow   = np.arange(nCities)                      # Cria um chute inicial
    valueNow  = calTourMileage(tourNow,nCities,distMat) # Calcula a distancia total
    tourBest  = tourNow.copy()                          # Melhor tour                   
    valueBest = valueNow                                # Melhor valor de score
    recordBest = []                                     # Salvar melhores scores
    recordNow  = []                                     # Scores atuais

    # Inicio do loop principal
    iter = 0    # Variavel auxiliar para iteracoes
    while tNow >= tFinal:   # Enquanto a temperatura estiver acima do minimo
        for k in range(nMarkov): # loop interno
            tourNew = mutateSwap(tourNow, nCities) # Pertuba o estado atual
            valueNew = calTourMileage(tourNew,nCities,distMat) # Novo score
            deltaE = valueNew - valueNow

            # Verifica de forma probabilistica se será aceito essa pertubação
            if deltaE < 0:                          # Se for melhor aceita de qualquer jeito
                accept = True
                if valueNew < valueBest:            # Verifica se a funcao otima é melhor e pega novo melhor
                    tourBest[:] = tourNew[:]
                    valueBest = valueNew
            else:                                   # Já que é pior vê se vale a pena
                pAccept = math.exp(-deltaE/tNow)    # Calculo probabilistico
                if pAccept > random.random():
                    accept = True
                else:
                    accept = False

            # Caso aceite pega os novos valores como atuais
            if accept == True:               
                tourNow[:] = tourNew[:]
                valueNow = valueNew

        # Inverte os caminhos
        tourNow = np.roll(tourNow,2)                # Muda o axis

        # Salva os melhores valores
        recordBest.append(valueBest)      
        recordNow.append(valueNow)             
        # print('i:{}, t(i):{:.2f}, valueNow:{:.1f}, valueBest:{:.1f}'.format(iter,tNow,valueNow,valueBest))

        # Aumenta a iteração e diminui a temperatura
        iter = iter + 1
        tNow = tNow * alfa   
    
    # Finaliza o Simulated Annelaing

    # Começa a plotagem

    figure1 = plt.figure()    
    num = len(tourBest)
    x0, y0 = coordinates[tourBest[num - 1]]
    x1, y1 = coordinates[tourBest[0]]
    plt.scatter(int(x0), int(y0), s=15, c='r')      
    plt.plot([x1, x0], [y1, y0], c='b')            
    for i in range(num - 1):
        x0, y0 = coordinates[tourBest[i]]
        x1, y1 = coordinates[tourBest[i + 1]]
        plt.scatter(int(x0), int(y0), s=15, c='r')  
        plt.plot([x1, x0], [y1, y0], c='b')      

    plt.xlabel("Distancia total: {:.1f}".format(valueBest))
    plt.title("Números de cidades: {:d}".format(num)) 
    plt.show()
    figure2 = plt.figure()    
    plt.title("Número de cidades: {:d}".format(nCities)) 
    plt.plot(np.array(recordBest),'b-', label='Melhor')         
    plt.plot(np.array(recordNow),'g-', label='Atual')           
    plt.xlabel("iter")                                      
    plt.ylabel("Distancia à percorrer")                               
    plt.legend()                                              
    plt.show()

    print("Melhor caminho: \n", tourBest)
    print("Melhor score: {:.1f}".format(valueBest))
    exit()



if __name__ == '__main__':
    ns = [10, 20, 50, 100, 250, 500, 1000, 5000, 10000]
    for n in ns:
        t = time.time()
        main(n)
        print(f'Resultado com {n} cidades é {time.time() - t}')
