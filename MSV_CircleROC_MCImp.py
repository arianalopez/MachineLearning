
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

clf = SVC(kernel='rbf')
clf.fit(X_train,np.ravel(Y_train))


xx, yy = np.meshgrid(np.linspace(-1.5, 1.5, 100),
                     np.linspace(-1.5, 1.5, 100))

Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

contours = plt.contour(xx, yy, Z, levels=[0], linewidths=5,
                       linetypes='--')
plt.scatter(X_train.iloc[:,0], X_train.iloc[:,1], s=20, c=Y_train.iloc[:,0], cmap=plt.cm.coolwarm,linewidths=0)
plt.xticks(())
plt.yticks(())
plt.axis([-1.5, 1.5, -1.5, 1.5])
plt.show()

#Definimos Y predecida
Y_pred=clf.predict(X_test)
print(Y_pred, Y_test)
#plt.scatter(X_test.iloc[:,0], X_test.iloc[:,1], s=20, c=Y_pred, cmap=plt.cm.coolwarm,linewidths=0)
#plt.xticks(())
#plt.yticks(())
#plt.axis([-1.5, 1.5, -1.5, 1.5])
#plt.show()

#Hacemos la matriz de confusion
y = np.array(Y_test)
class_names = np.unique(y)
print(class_names)

cmtrx = confusion_matrix(Y_test,Y_pred,labels=class_names)
labels = class_names
print("Confusion matrix:\n%s" % cmtrx)

print(pd.crosstab(pd.Series(Y_test), pd.Series(Y_pred), rownames=['True'], colnames=['Predicted'], margins=True))

norm_conf = []
for i in cmtrx:
    a = 0
    tmp_arr = []
    a = sum(i, 0)
    for j in i:
        tmp_arr.append(float(j)/float(a))
    norm_conf.append(tmp_arr)

fig = plt.figure()
plt.clf()
ax = fig.add_subplot(111)
ax.set_aspect(1)
res = ax.imshow(np.array(norm_conf), cmap=plt.cm.coolwarm,
                interpolation='nearest')

width = len(cmtrx)
height = len(cmtrx[0])

for x in xrange(width):
    for y in xrange(height):
        ax.annotate(str(cmtrx[x][y]), xy=(y, x),
                    horizontalalignment='center',
                    verticalalignment='center')

cb = fig.colorbar(res)
plt.title('Confusion Matrix')
plt.xticks(range(width), labels[:width])
plt.yticks(range(height), labels[:height])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('confusion_matrix.png', format='png')
plt.show()


# Graficamos la curva ROC
#fpr, tpr, thresholds = roc_curve(Y_test.iloc[:,0], Y_test.iloc[:,1])
fpr, tpr, thresholds = roc_curve(Y_test, Y_pred)
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
