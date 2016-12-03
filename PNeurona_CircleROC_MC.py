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
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix


nsim = 5000
X_1 = np.random.uniform(-1.5,1.5,nsim)
X_2 = np.random.uniform(-1.5,1.5,nsim)
X1_list = []
X2_list = []
R_list = []
XYs=[]

for i in range(len(X_1)):
    X1_list.append(X_1[i])
    X2_list.append(X_2[i])
    if (pow(X_1[i],2)+pow(X_2[i],2))< 1:
        rad =1
    else:
        rad =0
    R_list.append(rad)


XYdf = pd.DataFrame(zip(X1_list,X2_list,R_list),columns=["x1","x2","y"])

X_train, X_test, Y_train, Y_test = train_test_split(XYdf[['x1','x2']], XYdf[['y']], train_size=0.75)


net = buildNetwork(2,6,1)
ds = SupervisedDataSet(2, 1)
ds.setField('input',X_train)
ds.setField('target',Y_train)

X_test1 = np.ravel(X_test.iloc[0:,0:1])
X_test2 = np.ravel(X_test.iloc[0:,1:2])
#print(X_test,X1te[2],X2te[2])

trainer = BackpropTrainer(net, ds)
for i in range(20):
    print(i)
    trainer.train()

#New data for testing:
Y_testPred = []
Y_testR1 = []

for i in range(len(X_test)):
    r1 = net.activate([X_test1[i],X_test2[i]])
    if r1 >=0.5:
        r_cat = 1
    else:
        r_cat = 0
    Y_testPred.append(r_cat)
    Y_testR1.append(r1)
plt.scatter(X_test1, X_test2, c=Y_testPred, s=40,linewidths=0,cmap=plt.cm.coolwarm)
plt.show()

Y_test["Ypred"] = Y_testPred
#Y_test.groupby('y').mean()
print(Y_testPred)
print(Y_testR1)

#Matriz de Confusion
#Si Yreal=0 y Ypred=0 es TN
#Si Yreal=0 y Ypred<>0 es FP
#Si Yreal<>0 y Ypred=0 es FN
#Si Yreal<>0 y Ypred<>0 es TP

mtx_conf = []
for i in range(len(Y_test)):
    if Y_test["y"].iloc[i] == 0:
        if Y_test["Ypred"].iloc[i] == 0:
            mtx_conf.append("TN")
        else:
            mtx_conf.append("FP")
    else:
        if Y_test["Ypred"].iloc[i] == 0:
            mtx_conf.append("FN")
        else:
            mtx_conf.append("TP")

Y_test.loc[:,"mtx_conf"] = mtx_conf
Y_test.groupby("mtx_conf").count()["Ypred"]

conf = confusion_matrix(Y_test.iloc[:,0], Y_test.iloc[:,1])
cm_normalized = conf.astype('float') / conf.sum(axis=1)[:, np.newaxis]

#Graficamos la matriz de confusion

plt.scatter(X_test.iloc[:,0], X_test.iloc[:,1], c=Y_test.iloc[:,0], linewidths=0,cmap=plt.cm.seismic)
plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.coolwarm)
plt.title("Matriz de Confusion")
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ["Dentro","Fuera"], rotation=45)
plt.yticks(tick_marks, ["Dentro","Fuera"])
plt.tight_layout()
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Graficamos la curva ROC

fpr, tpr, thresholds = roc_curve(Y_test.iloc[:,0], Y_test.iloc[:,1])
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(fpr, tpr, lw=2, label='Roc Curve', color='gray')
plt.axis([-0.5, 1.05, -0.05, 1.05])
plt.title("Roc Curve", color='darkblue')
plt.plot([-.05, 1.5], [-.05, 1.5], '--', color=(.6, .6, .6), label='Barrera')
plt.xlabel('FP', color='red')
plt.ylabel('TP', color="red")
plt.legend(loc="upper left")
plt.show()

