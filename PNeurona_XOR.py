from matplotlib import pyplot
import matplotlib.pyplot as plt
import pybrain
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import random as rnd
import pandas as pd
import numpy as np
import csv
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator

Yr=[0,1,1,0]
x1=[0,0,1,1]
x2=[0,1,0,1]
XYs=[]

for k in range (100):
    for i in range(4):
        XYs.append([x1[i],x2[i],Yr[i]])

XYdf = pd.DataFrame(XYs,columns=['x1','x2','y'])
#print(XYdf.iloc[0:len(XYdf.index),0])
#print(XYdf.iloc[0:len(XYdf.index),1])

net = buildNetwork(2,8,1)

ds = SupervisedDataSet(2, 1)
ds.setField('input',XYdf[['x1','x2']])
ds.setField('target',XYdf[['y']])

x1a =XYdf[['x1']].values
x2a = XYdf[['x2']].values
y1a = XYdf[['y']].values

trainer = BackpropTrainer(net, ds)
for i in range(1000):
    trainer.train()

#ploteamos los puntos Y de nuestras X1 y X1
plt1 = plt.figure()
pltreal = plt1.add_subplot(111)
pltreal.scatter(x1a[:,0],x2a[:,0],color=['red' if i[0]==0 else 'blue' for i in y1a],linewidths=5)
plt.show()

#Sección de testing con nuevos datos (graphlab)
X_ax = 10000
X_1 = np.random.random_sample(X_ax,)
X_2 = np.random.random_sample(X_ax,)
X_1list = []
X_2list = []
Z = []
for i in range(len(X_1)):
    X_1list.append(X_1[i])
    X_2list.append(X_2[i])
    r1 = net.activate([X_1[i],X_2[i]])
    Z.append(r1)
cm = plt.cm.get_cmap('RdYlBu')
sc = plt.scatter(X_1list, X_2list, c=Z, vmin=0, vmax=1, s=40, cmap=cm,edgecolors='none')
plt.colorbar(sc)
plt.show()
