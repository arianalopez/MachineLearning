
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
from sklearn import tree
import pylab as pl
from matplotlib.ticker import MultipleLocator
from pandas_confusion import ConfusionMatrix
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier


df = pd.read_csv("abalonecsv.csv")
col_names = df.columns.tolist()
print (col_names)

to_show = col_names[:8] #+ col_names[-9:]
print "\nSample data:"
print(df[to_show].head(8))

X_train, X_test, Y_train, Y_test = train_test_split(df[["Lenght","Diameter","Height","Wweight","Shuweight","Vweight","Sheweight","Rings"]],df[["Sex"]], train_size=0.75)

###################DECISION TREE###########################33

clf = tree.DecisionTreeClassifier()
clf.fit(X_train, np.ravel(Y_train))
Y_pred = clf.predict(X_test)

#Creamos la matriz
y = np.array(Y_test)
class_names = np.unique(y)
print(class_names)
cmtrx = confusion_matrix(Y_test,Y_pred,labels=class_names)
labels = class_names
print("Confusion matrix Tree:\n%s" % cmtrx)

#Metrics de la matriz
#print(pd.DataFrame(Y_test).tail())
#print(pd.DataFrame(Y_pred).tail())
y_true = Y_test
y_predi = Y_pred
#print(pd.crosstab(y_true, y_predi, rownames=['True'], colnames=['Predicted'], margins=True))
print('classif_report tree',classification_report(y_true, y_predi))
print('accuracy_score tree',accuracy_score(y_true, y_predi))
MSE = mean_squared_error(Y_test, Y_pred)
print('MSE tree',MSE)


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


###################RANDOM FOREST###########################33

clf = RandomForestClassifier(n_estimators=10)
clf.fit(X_train, np.ravel(Y_train))
Y_pred = clf.predict(X_test)

#Creamos la matriz
y = np.array(Y_test)
class_names = np.unique(y)
print(class_names)
cmtrx = confusion_matrix(Y_test,Y_pred,labels=class_names)
labels = class_names
print("Confusion matrix RndForest:\n%s" % cmtrx)

#Metrics de la matriz
#print(pd.DataFrame(Y_test).tail())
#print(pd.DataFrame(Y_pred).tail())
y_true = Y_test
y_predi = Y_pred
#print(pd.crosstab(y_true, y_predi, rownames=['True'], colnames=['Predicted'], margins=True))
print('classif_report RndForst',classification_report(y_true, y_predi))
print('accuracy_score RndForst',accuracy_score(y_true, y_predi))
MSE = mean_squared_error(Y_test, Y_pred)
print('MSE RndForst',MSE)


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
plt.title('Confusion Matrix RndForst')
plt.xticks(range(width), labels[:width])
plt.yticks(range(height), labels[:height])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('confusion_matrix.png', format='png')
plt.show()



###################DECISION TREE ADABOOST###########################33

clf = AdaBoostClassifier(tree.DecisionTreeClassifier(),n_estimators=5000,learning_rate=1)
clf.fit(X_train, np.ravel(Y_train))
Y_pred = clf.predict(X_test)

#Creamos la matriz
y = np.array(Y_test)
class_names = np.unique(y)
print(class_names)
cmtrx = confusion_matrix(Y_test,Y_pred,labels=class_names)
labels = class_names
print("Confusion matrix AdBTree:\n%s" % cmtrx)

#Metrics de la matriz
#print(pd.DataFrame(Y_test).tail())
#print(pd.DataFrame(Y_pred).tail())
y_true = Y_test
y_predi = Y_pred
#print(pd.crosstab(y_true, y_predi, rownames=['True'], colnames=['Predicted'], margins=True))
print('classif_report AdBtree',classification_report(y_true, y_predi))
print('accuracy_score AdBtree',accuracy_score(y_true, y_predi))
MSE = mean_squared_error(Y_test, Y_pred)
print('MSE AdBtree',MSE)


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
plt.title('Confusion Matrix AdBTree')
plt.xticks(range(width), labels[:width])
plt.yticks(range(height), labels[:height])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('confusion_matrix.png', format='png')
plt.show()






'''
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
'''''