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

nsim = 1000
X_1 = np.random.uniform(-1.5,1.5,nsim)
Y_1 = np.random.uniform(-1.5,1.5,nsim)
X_list = []
Y_list = []
R_list = []
XYs=[]

for i in range(len(X_1)):
    X_list.append(X_1[i])
    Y_list.append(Y_1[i])
    if (pow(X_1[i],2)+pow(Y_1[i],2))< 1:
        rad =1
    else:
        rad =0
    R_list.append(rad)


for k in range (len(X_list)):
    XYs.append([X_list[k],Y_list[k],R_list[k]])

XYdf = pd.DataFrame(XYs,columns=['x1','x2','y'])

net = buildNetwork(2,6,1)

ds = SupervisedDataSet(2, 1)
ds.setField('input',XYdf[['x1','x2']])
ds.setField('target',XYdf[['y']])

x1a =XYdf[['x1']].values
x2a = XYdf[['x2']].values
y1a = XYdf[['y']].values

trainer = BackpropTrainer(net, ds)
for i in range(500):
    trainer.train()


#New data for testing:

nsim = 10000
X_1Test = np.random.uniform(-1.5,1.5,nsim)
Y_1Test = np.random.uniform(-1.5,1.5,nsim)
X_listTest = []
Y_listTest = []
R_listTest = []

for i in range(len(X_1Test)):
    X_listTest.append(X_1Test[i])
    Y_listTest.append(Y_1Test[i])
    r1 = net.activate([X_1Test[i],Y_1Test[i]])
    R_listTest.append(r1)
plt.scatter(X_listTest, Y_listTest, c=R_listTest, linewidths=0)
plt.show()

