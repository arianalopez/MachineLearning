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
from decimal import *
import numpy as np


df = pd.read_csv("regLinPoli.csv")
print(df)

X_train, X_test, Y_train, Y_test = train_test_split(df[['X','X2','X3','X4']],df[['y']], train_size=.75)

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
    return Ysal

def entrena(w0, W, X, y, nu, lam):
    for i in range(len(X)):
        Xa = X.ix[i]
        sal = salida(w0, W, Xa)
        error = y.ix[i] - sal
        w0 = w0 + (nu * error)
        for j in range(len(Xa)):
            W[j] = W[j] + (nu * error * X.ix[i][j])-(lam * W[j])
    return w0, W

nu = 0.01
w0 = (rnd.random())
W = [rnd.random(),rnd.random(),rnd.random(),rnd.random()]
lam = np.arange(0.001,0.01,0.001)

#Wentrena = entrena(w0, W, Xscaler, Yscaler, nu, lam)
#print("Wentrena",Wentrena)

#asi se ven los datos ploteados con 1 lambda
#Ygorro = float(Wentrena[0]) + (Xscaler[0] * float(Wentrena[1][0]))+(Xscaler[1] * float(Wentrena[1][1]))+(Xscaler[2] * float(Wentrena[1][2]))+(Xscaler[3] * float(Wentrena[1][3]))
#print(Ygorro)
#plt.scatter(Xscaler[0], Yscaler[0],color = "green")
#plt.plot(Xscaler[0], Ygorro, color = "red")
#plt.show()

errores = []
for i in range(len(lam)):
    Wentrena = entrena(w0, W, Xscaler, Yscaler, nu, lam[i])
    Ygorro = float(Wentrena[0]) + (Xscaler[0] * float(Wentrena[1][0]))+(Xscaler[1] * float(Wentrena[1][1]))+(Xscaler[2] * float(Wentrena[1][2]))+(Xscaler[3] * float(Wentrena[1][3]))
    Ygorro = pd.DataFrame(Ygorro)
    error = np.mean((Yscaler - Ygorro) ** 2)
    errores.insert(i, error)

print("errores",errores)

plt.plot(lam, errores, color = 'red')
plt.show()


resul_test = entrena(w0_i, W_i, Xscaler_p, Yscaler_p, nu_i, lam1[1])
resul_test