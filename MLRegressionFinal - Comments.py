import random
import sys
import os
import winpython
import pip
import sklearn
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import csv
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot
import matplotlib.pyplot as plt
import numpy as np
#%matplotlib inline

df = pd.read_csv("regLin.csv")
print (df)

#train_test_split: split arrays or matrices into random train and test subsets
#train_size: represent the proportion of the dataset to include in the train split
'''This is why a common practice in machine learning to evaluate an algorithm
is to split the data at hand into two sets, one that we call
the training set on which we learn data properties and one that we call
the testing set on which we test these properties. '''
X_train, X_test, Y_train, Y_test = train_test_split(df[["X"]],df[["y"]], train_size=0.75)
print(X_train,Y_train)

print("\n"*2)

''' We call our estimator instance clf, as it is a classifier.
It now must be fitted to the model, that is, it must learn from the model.
This is done by passing our training set to the fit method
La reg crea una fun en donde el error es el mas pequeno'''
clf = LinearRegression()
'''ajusta el modelo a los datos de entrenamiento'''
clf.fit(X_train, Y_train)
print ("clf ", clf)

print("\n"*2)

#
print("np mean", np.mean((clf.predict(X_test)-Y_test) ** 2))
print("coeficiente ",clf.coef_)
print("intercepto ", clf.intercept_)

plt.scatter(X_test,Y_test,color='black')
plt.plot(X_test,clf.predict(X_test),color='blue')
plt.show()

w0= clf.intercept_[0]
w1=clf.coef_[0][0]
W=range(-5,5,1)+w0
error=[]
for i in W:
    y_predecida=w1*X_test+i
    C=np.subtract(y_predecida, Y_test)
    C=C**2
    error.append(np.mean(C))
plt.plot(W,error)
plt.show()
