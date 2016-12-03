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


df = pd.read_csv("regLin.csv")
print (df)

#train_test_split: split arrays or matrices into random train and test subsets
#train_size: represent the proportion of the dataset to include in the train split
X_train, X_test, Y_train, Y_test = train_test_split(df[["X"]],df[["y"]], train_size=0.75)
print(X_train,Y_train)

print("\n"*2)

clf = LinearRegression()
clf.fit(X_train, Y_train)

print("coeficiente ",clf.coef_)
print("intercepto ", clf.intercept_)

print("\n"*2)

print("Error Y predic - Y real", np.mean((clf.predict(X_test)-Y_test) ** 2))

plt.scatter(X_test,Y_test,color='black')
plt.plot(X_test,clf.predict(X_test),color='blue')
plt.show()

w0= clf.intercept_[0]
w1=clf.coef_[0][0]
W=range(-10,10,1)+w0
error=[]
for i in W:
    y_predecida=w1*X_test+i
    C=np.subtract(y_predecida, Y_test)
    C=C**2
    error.append(np.mean(C))
plt.plot(W,error)
plt.show()



