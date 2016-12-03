
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
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
import math

#kaggle y UCI

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


XYdf = pd.DataFrame(zip(X_list,Y_list,R_list),columns=["x1","x2","y"])

X_train, X_test, Y_train, Y_test = train_test_split(XYdf[['x1','x2']], XYdf[['y']], train_size=0.75)

#test scatter
#plt.scatter(XYdf['x1'],XYdf['x2'], c=XYdf['y'], linewidths=0)
#plt.show()

clf = SVC(kernel='rbf')
clf.fit(X_train,np.ravel(Y_train))

xx, yy = np.meshgrid(np.linspace(-1.5, 1.5, 100),
                     np.linspace(-1.5, 1.5, 100))

Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

contours = plt.contour(xx, yy, Z, levels=[0], linewidths=5,
                       linetypes='--')
plt.scatter(XYdf['x1'], XYdf['x2'], s=20, c=XYdf['y'], cmap=plt.cm.coolwarm,linewidths=0)
plt.xticks(())
plt.yticks(())
plt.axis([-1.5, 1.5, -1.5, 1.5])
plt.show()

#Definimos Y predecida
Y_pred=clf.predict(X_test)
Y_test["Ypred"] = Y_pred
Y_test.groupby('y').mean()
print(Y_pred)

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


#predict.proba

conf = confusion_matrix(Y_test.iloc[:,0], Y_test.iloc[:,1])
cm_normalized = conf.astype('float') / conf.sum(axis=1)[:, np.newaxis]
print(conf)

#Graficamos la matriz de confusion

plt.scatter(X_test.iloc[:,0], X_test.iloc[:,1], c=Y_test.iloc[:,0], linewidths=0,cmap=plt.cm.seismic)
plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.coolwarm)
plt.title("Matriz de Confusion")
plt.colorbar(cmap=plt.cm.coolwarm)
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


'''
AQUI CALCULADO CON FUNCIONES DE SKLEARN
res1 = clf.decision_function(zip(X_test.iloc[:,0],X_test.iloc[:,1]))
fpr,tpr,umbrales = roc_curve(Y_test.iloc[:,0],res1)
print(fpr,tpr,umbrales)
print(auc(fpr,tpr))

plt1 = plt.figure()
pltreal = plt1.add_subplot(111)
plt.plot(fpr,tpr)
plt.axis([-.05,.2,.9,1.05])
#for x in range(0,len(Y_test)):
#    a = umbrales[x]
#    b = [fpr[x],tpr[x]]
#    pltreal.annotate('(%s)' % a,xy=b,size='small')
plt.show()
'''
