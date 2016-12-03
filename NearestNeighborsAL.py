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
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier


liminf = -1000
limsup = 1000

center1 = []
center2 = []
for i in range(5):
    c1 = np.random.uniform(liminf,limsup)
    c2 = np.random.uniform(liminf,limsup)
    center1.append(c1)
    center2.append(c2)

#print(center1,center2)
plt.scatter(center1, center2, s=20, cmap=plt.cm.cool)
plt.show()

nsim = 5000
X_1 = np.random.uniform(liminf,limsup,nsim)
X_2 = np.random.uniform(liminf,limsup,nsim)

y = []
R = 300

for i in range(len(X_1)):
    y1=0
    for j in range(len(center1)):
        if ((((X_1[i]-center1[j])**2)+((X_2[i]-center2[j])**2)) < R**2):
            y1 = 1
    y.append(y1)

XYdf = []
for k in range (len(X_1)):
    XYdf.append([X_1[k],X_2[k],y[k]])
XYdf = pd.DataFrame(XYdf,columns=['x1','x2','y'])

#print(XY)
plt.scatter(XYdf['x1'], XYdf['x2'], s=10, c=XYdf['y'], cmap=plt.cm.coolwarm)
plt.show()
X_train, X_test, Y_train, Y_test = train_test_split(XYdf[['x1','x2']], XYdf[['y']], train_size=0.75)
print(pd.DataFrame(Y_test).tail())
print(pd.DataFrame(X_test).tail())


##########################Tree##################################

clf = AdaBoostClassifier(tree.DecisionTreeClassifier(),n_estimators=5000,learning_rate=1)
clf.fit(X_train, Y_train)

Y_pred = clf.predict(X_test)

#np.set_printoptions(threshold=np.nan)
y = np.array(Y_test)
class_names = np.unique(y)
print(class_names)

cmtrx = confusion_matrix(Y_test,Y_pred,labels=class_names)
labels = class_names
print("Confusion matrix:\n%s" % cmtrx)

#Metrics
print(pd.DataFrame(Y_test).tail())
print(pd.DataFrame(Y_pred).tail())

y_true = Y_test
y_predi = Y_pred
print(pd.crosstab(y_true, y_predi, rownames=['True'], colnames=['Predicted'], margins=True))
print('classif_report',classification_report(y_true, y_predi))
print('accuracy_score',accuracy_score(y_true, y_predi))
MSE = mean_squared_error(Y_test, Y_pred)
print('MSE',MSE)


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
plt.title('Confusion Matrix Tree')
plt.xticks(range(width), labels[:width])
plt.yticks(range(height), labels[:height])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('confusion_matrix.png', format='png')
plt.show()


# Graficamos la curva ROC
fpr, tpr, thresholds = roc_curve(Y_test, Y_pred)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(fpr, tpr, lw=2, label='Roc Curve Tree', color='gray')
plt.axis([-0.5, 1.05, -0.05, 1.05])
plt.title("Roc Curve Tree", color='darkblue')
plt.plot([-.05, 1.5], [-.05, 1.5], '--', color=(.6, .6, .6), label='Barrera')
plt.xlabel('FP', color='red')
plt.ylabel('TP', color="red")
plt.legend(loc="upper left")
plt.show()



##########################NEAREST NEIGHBORS######################

neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(X_train,np.ravel(Y_train))
Y_pred = neigh.predict(X_test)
print(pd.DataFrame(Y_test).tail())
print(pd.DataFrame(Y_pred).tail())

#Hacemos la matriz de confusion
y = np.array(Y_test)
class_names = np.unique(y)
print(class_names)
cmtrx = confusion_matrix(Y_test,Y_pred,labels=class_names)
labels = class_names
print("Confusion matrix:\n%s" % cmtrx)

#Metrics
MSE = mean_squared_error(Y_test, Y_pred)
print('MSE',MSE)
y_true = Y_test
y_predi = Y_pred
print(pd.crosstab(y_true, y_predi, rownames=['True'], colnames=['Predicted'], margins=True))
print('classifn_report',classification_report(y_true, y_predi))
print('accuracy_score',accuracy_score(y_true, y_predi))

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
plt.title('Confusion Matrix NNS')
plt.xticks(range(width), labels[:width])
plt.yticks(range(height), labels[:height])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('confusion_matrix.png', format='png')
plt.show()


# Graficamos la curva ROC
fpr, tpr, thresholds = roc_curve(Y_test, Y_pred)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(fpr, tpr, lw=2, label='Roc Curve NNS', color='gray')
plt.axis([-0.5, 1.05, -0.05, 1.05])
plt.title("Roc Curve", color='darkblue')
plt.plot([-.05, 1.5], [-.05, 1.5], '--', color=(.6, .6, .6), label='Barrera')
plt.xlabel('FP', color='red')
plt.ylabel('TP', color="red")
plt.legend(loc="upper left")
plt.show()
