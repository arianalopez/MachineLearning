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


trainer = BackpropTrainer(net, ds)
for i in range(20):
    print(i)
    trainer.train()

#New data for testing:
X_test1 = np.ravel(X_test.iloc[0:,0:1])
X_test2 = np.ravel(X_test.iloc[0:,1:2])

Y_pred = []
Y_predR1 = []
umbral = 0.46

for i in range(len(X_test)):
    r1 = net.activate([X_test1[i],X_test2[i]])
    if r1 >=umbral:
        r_cat = 1
    else:
        r_cat = 0
    Y_pred.append(r_cat)
    Y_predR1.append(r1)
plt.scatter(X_test1, X_test2, c=Y_pred, s=40,linewidths=0,cmap=plt.cm.coolwarm)
plt.show()

print(pd.DataFrame(zip(Y_pred, Y_test, Y_predR1)))

# Graficamos la curva ROC

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

#Matriz de confusion

y = np.array(Y_test)
class_names = np.unique(y)
print(class_names)

cmtrx = confusion_matrix(Y_test,Y_pred,labels=class_names)
labels = class_names
print("Confusion matrix:\n%s" % cmtrx)

#print(pd.crosstab(pd.Series(Y_test), pd.Series(Y_pred), rownames=['True'], colnames=['Predicted'], margins=True))

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
