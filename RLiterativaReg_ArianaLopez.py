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


df = pd.read_csv("regLin.csv")
print(df)

X_train, X_test, Y_train, Y_test = train_test_split(df[["X"]],df[["y"]], train_size=.75)
X_scaler = preprocessing.StandardScaler().fit(X_train)
Y_scaler = preprocessing.StandardScaler().fit(Y_train)
Xscaler = X_scaler.transform(X_train)
Yscaler = Y_scaler.transform(Y_train)
Xscaler = pd.DataFrame(Xscaler)
Yscaler = pd.DataFrame(Yscaler)

print(Xscaler, Yscaler)

def salida (w0, W, Xa):
    Ysal = w0
    for k in range(len(W)):
        Ysal = Ysal + (W[k] * Xa[k])
    return Ysal

def entrena(w0, W, X, y, nu):
    for i in range(len(X)):
        Xa = X.ix[i]
        sal = salida(w0, W, Xa)
        error = y.ix[i] - sal
        w0 = w0 + (nu * error)
        for j in range(len(Xa)):
            W[j] = W[j] + (nu * error * X.ix[i][j])
    return w0, W

nu = 0.01
w0 = (rnd.random())
W = [rnd.random()]

print(entrena(w0, W, Xscaler, Yscaler, nu))

Ygorro = w0 + (Xscaler * W[0])
print(Ygorro)
plt.scatter(Xscaler[0], Yscaler[0])
plt.plot(Xscaler[0], Ygorro, color = "red")
plt.show() #para que se muestre la tabla