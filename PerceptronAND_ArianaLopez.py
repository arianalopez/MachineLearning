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


def salida (w0, W, Xa):
    Ysal = w0
    for k in range(len(W)):
        Ysal = Ysal + (W[k] * Xa[k])
    if Ysal > 0:
        return 1
    else:
        return 0

Y_estim = []
errores = []
Y_estimT = []

#entrenamos nuestro modelo para obtener W's optimas
def entrena(w0, W, X, y, nu):
    for i in range(len(X)):
        Xa = X[i]
        sal = salida(w0, W, Xa)
        Y_estim.append(sal)
        error = y[i] - sal
        errores.append(error)
        w0 = w0 + (nu * error)
        for j in range(len(Xa)):
            W[j] = W[j] + (nu * error * X[i][j])
    return w0, W


Yr=[0,0,0,1]
x1=[0,0,1,1]
x2=[0,1,0,1]
Xs=[]

for i in range(4):
    Xs.append([x1[i],x2[i]])

nu = 0.5
w0 = (rnd.random())
W = [rnd.random(),rnd.random()]

for j in range(10000):
    w0,W = entrena(w0, W, Xs, Yr, nu)

print (Xs)
print (Yr)
print(w0,W)

#usamos las W's entrenadas para el test
for i in range(len(Xs)):
    Ysal = w0
    for k in range(len(W)):
        Ysal = Ysal + (W[k] * Xs[i][k])
    if Ysal > 0:
        Ysal= 1
    else:
        Ysal= 0
    Y_estimT.append(Ysal)
print(Y_estimT)


#calculamos la linea de decision y la ploteamos con la recta con W's optimas
plt.scatter(x1,x2,c="blue")
plt.xlim(-.5,1.5)
plt.ylim(-.5,1.5)
z1 = np.linspace(-1,1.5,100)
plt.plot(z1, (-w0/W[1]) - ((z1*W[0])/W[1]), color='green', linewidth=3)
plt.show() #para que se muestre la tabla








