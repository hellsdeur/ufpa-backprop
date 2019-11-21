from random import random
from random import seed
import matplotlib.pyplot as plt
import math
import pandas as pd

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
    return 1.0 / (1.0 + math.exp(-ativacao))

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
    return saida * (1.0 - saida)

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
    vetor_de_erros = []
    vetor_de_epocas = []
    for epoca in range(n_epoca): # sgd para um numero de epocas
        vetor_de_epocas.append(epoca)
        sum_erro = 0
        for datarow in treino: # feedforward, backprop e atualiza os pesos
            saidas = feedforward(rede, datarow)
            esperado = [0 for i in range(n_out)]
            esperado[datarow[-1]] = 1
            sum_erro += sum([(esperado[i] - saidas[i])**2 for i in range(len(esperado))])
            back_prop(rede, esperado)
            atualizar_pesos(rede, datarow, taxa_apre)
        vetor_de_erros.append(sum_erro)
        print('EPOCA = %d, TAXA_APRE = %.3f, ERRO = %.3f' % (epoca, taxa_apre, sum_erro))
    fig, ax = plt.subplots()
    plt.grid()
    plt.title('Evolucao do MSE ao longo das epocas')
    plt.ylabel('Erro Medio Quadratico')
    plt.xlabel('Epocas')
    ax.plot(vetor_de_epocas, vetor_de_erros, color='red')
        
def validacao(rede, datarow):
    saidas = feedforward(rede, datarow)
    return saidas.index(max(saidas)) # em saidas, retorna o valor maximo para a classe
        
def minmax_data(dataset):
    return [[min(coluna), max(coluna)] for coluna in zip(*dataset)]

def normalizar_dataset(dataset, minmax):
    for linha in dataset:
        for i in range(len(linha) - 1):
            linha[i] = (linha[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
    return dataset

def rodar_rede(dataset, n_hid,taxa_apre,epocas):
    minmax = minmax_data(dataset)
    datatreino = normalizar_dataset(dataset, minmax)
    datavalid = normalizar_dataset(dataset, minmax)
    datatreino = dataset[0:10]
    datavalid = dataset[10:20]
    
    n_inp = len(datatreino[0]) - 1
    n_out = len(set([datarow[-1] for datarow in datatreino]))
    seed(1)
    rede = inicializar_rede(n_inp,n_hid,n_out)
    treinar_rede(rede, datatreino, taxa_apre, epocas, n_out)
    for camada in rede:
        print(camada)
    for datarow in datavalid:
        valid = validacao(rede, datarow)
        print('ESPERADO = %d, OBTIDO = %d' % (datarow[-1], valid))
    
dataset = pd.read_csv('dadosmamografia.csv')
dataset = dataset.values.tolist()

n_hid = int(input('Numero de neuronios na camada escondida: '))
taxa_apre = float(input('Taxa de aprendizado: '))
epocas = int(input('Numero de epocas: '))

rodar_rede(dataset, n_hid, taxa_apre, epocas)