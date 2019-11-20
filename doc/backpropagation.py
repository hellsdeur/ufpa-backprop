from random import random
from random import seed
import math
# inicializando rede com pesos aleatorios
def inicializar_rede(n_inp, n_hid, n_out):
    rede = []
    
    # cada neuronio da escondida possui n_inp entradas + 1 bias
    camada_hid = [{'pesos': [random() for i in range(n_inp + 1)]} for i in range(n_hid)]
    rede.append(camada_hid)
    
    # cada neuronio da saida possui n_hid entradas + 1 bias
    camada_out = [{'pesos': [random() for i in range(n_hid + 1)]} for i in range(n_out)]
    rede.append(camada_out)
    
    # a rede eh um array de camadas, e cada camada um dicionario
    return rede

# calcular ativacao do neuronio para uma entrada
def ativacao(pesos, entradas):
    ativacao = pesos[-1] # adiciona o bias
    for i in range(len(pesos) - 1): # soma ponderada das entradas
        ativacao += pesos[i] * entradas[i]
    return ativacao

# funcao de ativacao: sigmoide
def transferencia(ativacao):
    #return 1.0 / (1.0 + math.exp(-ativacao))
    return  (2.0 / (1.0 + math.exp(-2 * ativacao))) - 1

# alimentacao adiante, obtendo vetor de saida
def feedforward(rede, datarow):
    entradas = datarow # a entrada da rede contem uma linha do dataset
    for camada in rede: # calcular a saida para cada camada
        prox_entrada = [] # saida de uma camada -> entrada da proxima
        for neuronio in camada: # calcular a saida para cada neuronio
            vetor_ativacao = ativacao(neuronio['pesos'], entradas)
            neuronio['saida'] = transferencia(vetor_ativacao)
            prox_entrada.append(neuronio['saida'])
        entradas = prox_entrada # para proxima camada
    return entradas # vetor com a saida da camada de saida

# calcular inclinacao da saida
def derivada(saida):
    #return saida * (1.0 - saida)
    return 1.0 - (saida**2)

# calcula o erro para cada camada e retropropaga
def back_prop(rede, esperado):
    # calculando o erro de tras pra frente
    for i in reversed(range(len(rede))):
        camada = rede[i]
        erros = []
        if i != len(rede) - 1: # se nao for a ultima camada
            for j in range(len(camada)):
                erro = 0.0
                for neuronio in rede[i + 1]: # erro = peso * erro da saida
                    erro += (neuronio['pesos'][j] * neuronio['delta'])
                erros.append(erro)
        else: # se for a camada de saida
            for j in range(len(camada)): # erro = esperado - saida
                neuronio = camada[j]
                erros.append(esperado[j] - neuronio['saida'])
        for j in range(len(camada)):
            neuronio = camada[j]
            neuronio['delta'] = erros[j] * derivada(neuronio['saida'])
            
# atualiza pesos a partir dos erros
def atualizar_pesos(rede, datarow, taxa_apre):
    for i in range(len(rede)):
        entradas = datarow[:-1]
        if i != 0:
            entradas = [neuronio['saida'] for neuronio in rede[i-1]]
        for neuronio in rede[i]:
            for j in range(len(entradas)):
                neuronio['pesos'][j] += taxa_apre * neuronio['delta'] * entradas[j]
            neuronio['pesos'][-i] += taxa_apre * neuronio['delta']
            
# treino da rede
def treinar_rede(rede, treino, taxa_apre, n_epoca, n_out):
    for epoca in range(n_epoca): # sgd para um numero de epocas
        sum_erro = 0
        for datarow in treino: # feedforward, backprop e atualiza os pesos
            saidas = feedforward(rede, datarow)
            esperado = [0 for i in range(n_out)]
            esperado[datarow[-1]] = 1
            sum_erro += sum([(esperado[i] - saidas[i])**2 for i in range(len(esperado))])
            back_prop(rede, esperado)
            atualizar_pesos(rede, datarow, taxa_apre)
            print('EPOCA = %d, TAXA_APRE = %.3f, ERRO = %.3f' % (epoca, taxa_apre, sum_erro))
        
# predicao
def predicao(rede, datarow):
    saidas = feedforward(rede, datarow)
    return saidas.index(max(saidas)) # em saidas, retorna o valor maximo para a classe
        
def minmaxdataset(dataset):
    return [[min(coluna), max(coluna)] for coluna in zip(*dataset)]

def normalizar_dataset(dataset, minmax):
    for linha in dataset:
        for i in range(len(linha) - 1):
            linha[i] = (linha[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

def rodar_rede(dataset,n_hid,taxa_apre,epocas):
    minmax = minmaxdataset(dataset)
    normalizar_dataset(dataset, minmax)
    n_inp = len(dataset[0]) - 1
    n_out = len(set([datarow[-1] for datarow in dataset]))
    seed(1)
    rede = inicializar_rede(n_inp,n_hid,n_out)
    treinar_rede(rede, dataset, taxa_apre, epocas, n_out)
    for camada in rede:
        print(camada)
    for datarow in dataset:
        pred = predicao(rede, datarow)
        print('ESPERADO = %d, OBTIDO = %d' % (datarow[-1], pred))
    
dataset = [[5,67,3,5,3,1],
            [5,58,4,5,3,1],
            [4,28,1,1,3,0],
            [5,57,1,5,3,1],
            [5,76,1,4,3,1],
            [3,42,2,1,3,1],
            [4,36,3,1,2,0],
            [4,60,2,1,2,0],
            [4,54,1,1,3,0],
            [3,52,3,4,3,0],
            [4,59,2,1,3,1],
            [4,54,1,1,3,1],
            [5,56,4,3,1,1],
            [5,42,4,4,3,1],
            [4,59,2,4,3,1],
            [5,75,4,5,3,1],
            [5,45,4,5,3,1],
            [5,55,4,4,3,0],
            [4,46,1,5,2,0],
            [5,54,4,4,3,1],
            [5,57,4,4,3,1],
            [4,39,1,1,2,0],
            [4,81,1,1,3,0],
            [4,60,2,1,3,0],
            [5,67,3,4,2,1],
            [4,55,3,4,2,0],
            [4,78,1,1,1,0],
            [4,50,1,1,3,0],
            [5,62,3,5,2,1],
            [5,64,4,5,3,1],
            [5,67,4,5,3,1],
            [4,74,2,1,2,0],
            [5,80,3,5,3,1],
            [4,49,2,1,1,0],
            [5,52,4,3,3,1],
            [5,60,4,3,3,1],
            [4,57,2,5,3,0],
            [5,74,4,4,3,1],
            [4,49,1,1,3,0],
            [4,45,2,1,3,0],
            [4,64,2,1,3,0],
            [4,73,2,1,2,0],
            [5,68,4,3,3,1],
            [5,52,4,5,3,0],
            [5,66,4,4,3,1],
            [4,25,1,1,3,0],
            [5,74,1,1,2,1],
            [4,64,1,1,3,0],
            [5,60,4,3,2,1],
            [5,67,2,4,1,0],
            [4,67,4,5,3,0],
            [5,44,4,4,2,1],
            [3,68,1,1,3,1],
            [5,58,4,4,3,1],
            [5,62,1,5,3,1],
            [4,73,3,4,3,1],
            [4,80,4,4,3,1],
            [5,59,2,1,3,1],
            [5,54,4,4,3,1],
            [5,62,4,4,3,0],
            [4,33,2,1,3,0],
            [4,57,1,1,3,0],
            [4,45,4,4,3,0],
            [5,71,4,4,3,1],
            [5,59,4,4,2,0],
            [4,56,1,1,3,0],
            [4,57,2,1,2,0],
            [5,55,3,4,3,1],
            [5,84,4,5,3,0],
            [5,51,4,4,3,1],
            [4,24,2,1,2,0],
            [4,66,1,1,3,0],
            [5,33,4,4,3,0],
            [4,59,4,3,2,0],
            [5,40,4,5,3,1],
            [5,67,4,4,3,1],
            [5,75,4,3,3,1],
            [5,86,4,4,3,0],
            [5,66,4,4,3,1],
            [5,46,4,5,3,1],
            [4,59,4,4,3,1],
            [5,65,4,4,3,1],
            [4,53,1,1,3,0],
            [5,67,3,5,3,1],
            [5,80,4,5,3,1],
            [4,55,2,1,3,0],
            [4,47,1,1,2,0],
            [5,62,4,5,3,1],
            [5,63,4,4,3,1],
            [4,71,4,4,3,1],
            [4,41,1,1,3,0],
            [5,57,4,4,4,1],
            [5,71,4,4,4,1],
            [4,66,1,1,3,0],
            [4,47,2,4,2,0],
            [3,34,4,4,3,0],
            [4,59,3,4,3,0],
            [5,67,4,4,3,1],
            [4,41,2,1,3,0],
            [4,23,3,1,3,0],
            [4,42,2,1,3,0],
            [5,87,4,5,3,1],
            [4,68,1,1,3,1],
            [4,64,1,1,3,0],
            [5,54,3,5,3,1],
            [5,86,4,5,3,1],
            [4,21,2,1,3,0],
            [4,53,4,4,3,0],
            [4,44,4,4,3,0],
            [4,54,1,1,3,0],
            [5,63,4,5,3,1],
            [4,45,2,1,2,0],
            [5,71,4,5,3,0],
            [5,49,4,4,3,1],
            [4,49,4,4,3,0],
            [5,66,4,4,4,0],
            [4,19,1,1,3,0],
            [4,35,1,1,2,0],
            [5,74,4,5,3,1],
            [5,37,4,4,3,1],
            [5,81,3,4,3,1],
            [5,59,4,4,3,1],
            [4,34,1,1,3,0],
            [5,79,4,3,3,1],
            [5,60,3,1,3,0],
            [4,41,1,1,3,1],
            [4,50,1,1,3,0],
            [5,85,4,4,3,1],
            [4,46,1,1,3,0],
            [5,66,4,4,3,1],
            [4,73,3,1,2,0],
            [4,55,1,1,3,0],
            [4,49,2,1,3,0],
            [3,49,4,4,3,0],
            [4,51,4,5,3,1],
            [2,48,4,4,3,0],
            [4,58,4,5,3,0],
            [5,72,4,5,3,1],
            [4,46,2,3,3,0],
            [4,43,4,3,3,1],
            [4,46,1,1,1,0],
            [4,69,3,1,3,0],
            [5,43,2,1,3,1],
            [5,76,4,5,3,1],
            [4,46,1,1,3,0],
            [4,59,2,4,3,0],
            [4,57,1,1,3,0],
            [3,45,2,1,3,0],
            [3,43,2,1,3,0],
            [4,45,2,1,3,0],
            [5,57,4,5,3,1],
            [5,79,4,4,3,1],
            [5,54,2,1,3,1],
            [4,40,3,4,3,0],
            [5,63,4,4,3,1],
            [4,52,2,1,3,0],
            [4,38,1,1,3,0],
            [3,72,4,3,3,0],
            [5,80,4,3,3,1],
            [5,76,4,3,3,1],
            [4,62,3,1,3,0],
            [5,64,4,5,3,1],
            [5,42,4,5,3,0],
            [4,64,4,5,3,0],
            [4,63,4,4,3,1],
            [4,24,2,1,2,0],
            [5,72,4,4,3,1],
            [4,63,2,1,3,0],
            [4,46,1,1,3,0],
            [3,33,1,1,3,0],
            [5,76,4,4,3,1],
            [4,36,2,3,3,0],
            [4,40,2,1,3,0],
            [5,58,1,5,3,1],
            [4,43,2,1,3,0],
            [3,42,1,1,3,0],
            [4,32,1,1,3,0],
            [5,57,4,4,2,1],
            [4,37,1,1,3,0],
            [4,70,4,4,3,1],
            [5,56,4,2,3,1],
            [5,73,4,4,3,1],
            [5,77,4,5,3,1],
            [5,67,4,4,1,1],
            [5,71,4,3,3,1],
            [5,65,4,4,3,1],
            [4,43,1,1,3,0],
            [4,40,2,1,3,0],
            [4,49,2,1,3,0],
            [5,76,4,2,3,1],
            [4,55,4,4,3,0],
            [5,72,4,5,3,1],
            [3,53,4,3,3,0],
            [5,75,4,4,3,1],
            [5,61,4,5,3,1],
            [5,67,4,4,3,1],
            [5,55,4,2,3,1],
            [5,66,4,4,3,1],
            [2,76,1,1,2,0],
            [4,57,4,4,3,1],
            [5,71,3,1,3,0],
            [5,70,4,5,3,1],
            [4,63,2,1,3,0],
            [5,40,1,4,3,1],
            [4,41,1,1,3,0],
            [4,47,2,1,2,0],
            [4,68,1,1,3,1],
            [4,64,4,3,3,1],
            [4,73,4,3,3,0],
            [4,39,4,3,3,0],
            [5,55,4,5,4,1],
            [5,53,3,4,4,0],
            [5,66,4,4,3,1],
            [4,43,3,1,2,0],
            [5,44,4,5,3,1],
            [4,77,4,4,3,1],
            [4,62,2,4,3,0],
            [5,80,4,4,3,1],
            [4,33,4,4,3,0],
            [4,50,4,5,3,1],
            [5,46,4,4,3,1],
            [5,49,4,5,3,1],
            [4,53,1,1,3,0],
            [3,46,2,1,2,0],
            [4,57,1,1,3,0],
            [4,54,3,1,3,0],
            [2,49,2,1,2,0],
            [4,47,3,1,3,0],
            [4,40,1,1,3,0],
            [4,45,1,1,3,0],
            [4,50,4,5,3,1],
            [5,54,4,4,3,1],
            [4,67,4,1,3,1],
            [4,77,4,4,3,1],
            [4,66,4,3,3,0],
            [4,36,2,3,3,0],
            [4,69,4,4,3,0],
            [4,48,1,1,3,0],
            [4,64,4,4,3,1],
            [4,71,4,2,3,1],
            [5,60,4,3,3,1],
            [4,24,1,1,3,0],
            [5,34,4,5,2,1],
            [4,79,1,1,2,0],
            [4,45,1,1,3,0],
            [4,37,2,1,2,0],
            [4,42,1,1,2,0],
            [4,72,4,4,3,1],
            [5,60,4,5,3,1],
            [5,85,3,5,3,1],
            [4,51,1,1,3,0],
            [5,54,4,5,3,1],
            [5,55,4,3,3,1],
            [4,64,4,4,3,0],
            [5,67,4,5,3,1],
            [5,75,4,3,3,1],
            [5,87,4,4,3,1],
            [4,46,4,4,3,1],
            [5,46,4,3,3,1],
            [5,61,1,1,3,1],
            [4,44,1,4,3,0],
            [4,32,1,1,3,0],
            [4,62,1,1,3,0],
            [5,59,4,5,3,1],
            [4,61,4,1,3,0],
            [5,78,4,4,3,1],
            [5,42,4,5,3,0],
            [4,45,1,2,3,0],
            [5,34,2,1,3,1],
            [4,27,3,1,3,0],
            [4,43,1,1,3,0],
            [5,83,4,4,3,1],
            [4,36,2,1,3,0],
            [4,37,2,1,3,0],
            [4,56,3,1,3,1],
            [5,55,4,4,3,1],
            [4,88,4,4,3,1],
            [5,71,4,4,3,1]]
            

n_hid = int(input('Numero de neuronios na camada escondida: '))
taxa_apre = float(input('Taxa de aprendizado: '))
epocas = int(input('Numero de epocas: '))

rodar_rede(dataset, n_hid, taxa_apre, epocas)