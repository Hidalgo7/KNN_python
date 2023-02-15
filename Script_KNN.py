import numpy as np
import pandas as pd
import statistics as st
import math
import csv
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import random
from sklearn.model_selection import KFold

def main():
    path_dataset = "mtcars.csv" # Escoged bien la ruta!!
    mtcars = pd.read_csv(path_dataset) # Leemos el csv
    # Discretizamos la variable clase para convertirlo en un problema de clasificacion
    ix_consumo_alto = mtcars.mpg >= 21
    mtcars.mpg[ix_consumo_alto] = 1
    mtcars.mpg[~ix_consumo_alto] = 0
    print("Este es el dataset sin normalizar")
    print(mtcars)
    print("\n\n")
    # Ahora normalizamos los datos
    mtcars_normalizado = mtcars.loc[:, mtcars.columns != 'mpg'].apply(normalize, axis=1)
    # Añadimos la clase a nuestro dataset normalizado
    mtcars_normalizado['mpg'] = mtcars['mpg']
    print("Este es el dataset normalizado")
    print(mtcars_normalizado)
    print("\n\n")
    # Hacemos un split en train y test con un porcentaje del 0.75 Train
    train,test = splitTrainTest(mtcars_normalizado,0.75)
    kFoldCV(mtcars_normalizado,4)
    # Separamos las labels del Test. Es como si no nos las dieran!!
    
    # Predecimos el conjunto de test
    true_labels = test.pop('mpg').tolist()
    acc = []
    for K in range (1,21):
        predicted_labels = []
        for index in test.index:
            predicted_labels.append(knn(test.loc[index],train,K))
            
        test['mpg'] = true_labels
        test['pred_mpg'] = predicted_labels
        print(test)
        test.pop('pred_mpg')
        test.pop('mpg')
        # Mostramos por pantalla el Accuracy por ejemplo
        print("Accuracy conseguido: {}".format(accuracy(true_labels, predicted_labels)))
        acc.append(accuracy(true_labels, predicted_labels))
        
    plt.plot(range(1,21),acc)
    

    # Algun grafico? Libreria matplotlib.pyplot
    return(0)

# FUNCIONES de preprocesado
def normalize(x):
    return((x-min(x)) / (max(x) - min(x)))

def standardize(x):
    return((x-st.mean(x))/st.variance(x))

# FUNCIONES de evaluacion
def splitTrainTest(data, percentajeTrain):
    """
    Takes a pandas dataframe and a percentaje (0-1)
    Returns both train and test sets
    """
    v  = np.random.rand(len(data))
    mascara = v > 0.75
    
    test = data.loc[mascara]
    train = data.loc[~mascara]
    print("TRAIN:")
    print(train)
    print("TEST:")
    print(test)
    return(train,test)

def kFoldCV(data, K):
    """
    Takes a pandas dataframe and the number of folds of the CV
    YOU CAN USE THE sklearn KFold function here
    How to: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """
    kf = KFold(n_splits = K, shuffle = True, random_state = 2)
    result = list(kf.split(data))
    print(result)

# FUNCION modelo prediccion
def knn(newx, data, K):
    """
    Receives two pandas dataframes. Newx consists on a single row df.
    Returns the prediction for newx
    """
    
    newx_list = newx.values.tolist()
    data_list = data.values.tolist()
    
    distances = []
    
    for case in data_list:
        distances.append(euclideanDistance2points(case,newx_list))
    
    distance_order = sorted(range(len(distances)), key=lambda k: distances[k])
    neighbors = []
    for i in range(0,K):
        neighbors.append(distance_order.index(i))
    
    labels = {}
    for x in neighbors:
        if str(data.loc[data.index[x]]['mpg']) in labels:
            labels[str(data.loc[data.index[x]]['mpg'])] += 1
        else:
            labels[str(data.loc[data.index[x]]['mpg'])] = 1
    newlabel = float(max(labels,key=labels.get))
    
    return(newlabel)

def euclideanDistance2points(x,y):
    """
    Takes 2 matrix - Not pandas dataframe!
    """
    sum = 0
    for (i,j) in zip(x,y):
        sum += (i-j)**2
    
    dist = math.sqrt(sum)
            
    return(dist)

# FUNCION accuracy
def accuracy(true, pred):
    cont = 0
    for (t,p) in zip(true,pred):
        if t == p:
            cont += 1
    acc = cont/len(true)
    return(acc)

if __name__ == '__main__':
    
    np.random.seed(100)
    main()