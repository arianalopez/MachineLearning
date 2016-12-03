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
print (df)

X_train, X_test, Y_train, Y_test = train_test_split(df[["X"]],df[["y"]], train_size=.75)
Y_tra = Y_train['y'].values.tolist()
X_tra = X_train['X'].values.tolist()
Y_tes = Y_test['y'].values.tolist()
X_tes = X_test['X'].values.tolist()

scaleX = preprocessing.StandardScaler()
scaleY = preprocessing.StandardScaler()
scaleX.fit(X_tra)
X_train=scaleX.transform(X_tra)
scaleY.fit(Y_tra)
Y_train=scaleY.transform(Y_tra)

scaleX.fit(X_tes)
X_train=scaleX.transform(X_tes)
scaleY.fit(Y_tes)
Y_train=scaleY.transform(Y_tes)

nu = .0001
numX = len(X_tra)

ListWo= []
ListW= []
ListYest= []
ListErrora= []
ListYpred = []

w = rnd.random()
w0 = rnd.random()

clf = LinearRegression()

for j in range(numX):
    w_sum = (w * X_tra[j])
    y_est = w_sum + w0
    errora = Y_tra[j] - y_est
    print(w0,w,y_est,errora)
    ListWo.append(w0)
    ListW.append(w)
    ListYest.append(y_est)
    ListErrora.append(errora)
    w = w + nu*(errora*X_tra[j])
    w0 = w0 + nu*errora

print('\n')
print("y_est",y_est)
print("wo",w0)
print("w1",w)
print("error",errora)


print('\n')
for i in range(numX):
    Ypred = w0+ w*X_tra[i]
    ListYpred.append(Ypred)


plt.scatter(X_tra,Y_tra,color='red')
plt.scatter(X_tes,Y_tes,color='black')
plt.scatter(X_tra,ListErrora,color="yellow")
plt.scatter(X_tra,ListYpred,color='darkblue')
plt.show()

#plt.scatter(X_tra,Ypred,color='blue')
#plt.show()