import random
from random import sample
import random as rnd
import sys
import os
import winpython
import pip
import sklearn
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import csv
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from matplotlib import pyplot
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv("regLin4.csv")
print(df)

X_train, X_test, Y_train, Y_test = train_test_split(df[["X"]],df[["y"]], train_size=.75)
X_scaler = preprocessing.StandardScaler().fit(X_train)
Y_scaler = preprocessing.StandardScaler().fit(Y_train)
Xscaler = X_scaler.transform(X_train)
Yscaler = Y_scaler.transform(Y_train)
Xscaler = pd.DataFrame(Xscaler)
Yscaler = pd.DataFrame(Yscaler)

X_scalerTe = preprocessing.StandardScaler().fit(X_test)
Y_scalerTe = preprocessing.StandardScaler().fit(Y_test)
XscalerTe = X_scalerTe.transform(X_test)
YscalerTe = Y_scalerTe.transform(Y_test)
XscalerTe = pd.DataFrame(XscalerTe)
YscalerTe = pd.DataFrame(YscalerTe)

print(Xscaler, Yscaler)

def salida (w0, W, Xa):
    Ysal = w0
    for k in range(len(W)):
        Ysal = Ysal + (W[k] * Xa[k])
    if Ysal >=0:
        return 1
    else:
        return 0

Y_estim = []
errores = []
Y_estimTe = []

#entrenamos nuestro modelo para obtener W's optimas
def entrena(w0, W, X, y, nu):
    for i in range(len(X)):
        Xa = X.values[i]
        sal = salida(w0, W, Xa)
        Y_estim.append(sal)
        error = y.values[i] - sal
        errores.append(error)
        w0 = w0 + (nu * error)
        for j in range(len(Xa)):
            W[j] = W[j] + (nu * error * X.values[i][j])
    print(w0,W)
    return w0, W

nu = 0.05
w0 = (rnd.random())
W = [rnd.random()]

w0,W = entrena(w0, W, Xscaler, Y_train, nu)

#usamos las W's entrenadas para el test
for i in range(len(XscalerTe)):
    YsalTest = w0 + (XscalerTe.values[i] * W[0])
    if YsalTest >0:
        YsalTest =1
    else:
        YsalTest= 0
    Y_estimTe.append(YsalTest)

#calculamos la linea de decision y la ploteamos con la Y estimada
vline1 = (0-w0/W[0])
plt.scatter(XscalerTe[0], Y_test, color = "blue") #estos son los originales
plt.scatter(XscalerTe[0], Y_estimTe, color = "red") #estos son los estimados
plt.axvline(vline1, color = "green",linewidth=2) #la linea de decision
plt.show() #para que se muestre la tabla




