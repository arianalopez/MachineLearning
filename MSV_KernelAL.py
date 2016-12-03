from matplotlib import pyplot
import matplotlib.pyplot as plt
import pybrain
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.svm import SVC
import random as rnd
import pandas as pd
import numpy as np
import csv
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator

df = pd.read_csv("andSVM.csv")

x = df[['X1','X2']]
Vy = df[['y']]
xarray = np.asarray(x)
yarray = np.asarray(Vy)

clf = SVC(C=20,kernel='linear')
clf.fit(xarray,np.ravel(yarray))

w0 = clf.intercept_[0]
w1 = clf.coef_[0][0]
w2 = clf.coef_[0][1]

print(w0,w1,w2)
'''
x2 = w0/w2 -w1/2*x1
'''
x_ax = np.linspace(-0.5,3,50)
m= -w1/w2
x2 = -w0/w2+m*x_ax

supportlow = clf.support_vectors_[0]
supporthigh = clf.support_vectors_[-1]

down = x_ax*m+(supportlow[1]-supportlow[0]*m)
up = x_ax*m+(supporthigh[1]-supporthigh[0]*m)


plt.scatter(df[['X1']],df[['X2']], c=yarray)
plt.plot(x_ax,down,linewidth=1, color='blue')
plt.plot(x_ax,up,linewidth=1, color='blue')
plt.plot(x_ax, (-w0/w2) - ((x_ax*w1)/w2), linewidth=3, color='red')
plt.show() #para que se muestre la tabla

