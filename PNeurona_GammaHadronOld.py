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
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from pybrain.structure import SigmoidLayer

XYdf = pd.read_csv("cherenkovGHray.csv")
print(XYdf)

X_train, X_test, Y_train, Y_test = train_test_split(XYdf[['fLength','fWidth','fSize','fConc','fConc1','fAsym','fM3Long','fM3Trans','fAlpha','fDist']], XYdf[['class']], train_size=0.75)

X_scaler = preprocessing.StandardScaler().fit(X_train)
Y_scaler = preprocessing.StandardScaler().fit(Y_train)
Xscaler = X_scaler.transform(X_train)
Yscaler = Y_scaler.transform(Y_train)
X_scalerTe = preprocessing.StandardScaler().fit(X_test)
Y_scalerTe = preprocessing.StandardScaler().fit(Y_test)
XscalerTe = X_scalerTe.transform(X_test)
YscalerTe = Y_scalerTe.transform(Y_test)
#train X and y
Xscaler = pd.DataFrame(Xscaler)
Yscaler = pd.DataFrame(Yscaler)
#test X and Y
XscalerTe = pd.DataFrame(XscalerTe)
YscalerTe = pd.DataFrame(YscalerTe)


net = buildNetwork(10,50,1,outclass=SigmoidLayer)
ds = SupervisedDataSet(10, 1)
ds.setField('input',Xscaler)
ds.setField('target',Y_train)


trainer = BackpropTrainer(net, ds)
for i in range(10):
    print(i)
    trainer.train()

#New data for testing:
X_test0 = np.ravel(XscalerTe.iloc[0:,0:1])
X_test1 = np.ravel(XscalerTe.iloc[0:,1:2])
X_test2 = np.ravel(XscalerTe.iloc[0:,2:3])
X_test3 = np.ravel(XscalerTe.iloc[0:,3:4])
X_test4 = np.ravel(XscalerTe.iloc[0:,4:5])
X_test5 = np.ravel(XscalerTe.iloc[0:,5:6])
X_test6 = np.ravel(XscalerTe.iloc[0:,6:7])
X_test7 = np.ravel(XscalerTe.iloc[0:,7:8])
X_test8 = np.ravel(XscalerTe.iloc[0:,8:9])
X_test9 = np.ravel(XscalerTe.iloc[0:,9:10])

Y_pred = []
Y_predR1 = []
umbral = 0.5

for i in range(len(X_test)):
    r1 = net.activate([X_test0[i],X_test1[i],X_test2[i],X_test3[i],X_test4[i],X_test5[i],X_test6[i],X_test7[i],X_test8[i],X_test9[i]])
    if r1 >=umbral:
        r_cat = 1
    else:
        r_cat = 0
    Y_pred.append(r_cat)
    Y_predR1.append(r1)
#plt.scatter(X_test1, X_test2, c=Y_pred, s=40,linewidths=0,cmap=plt.cm.coolwarm)
#plt.show()


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

#Metrics
print(pd.DataFrame(Y_test).tail())
print(pd.DataFrame(Y_pred).tail())
print(pd.DataFrame(Y_predR1).tail())
y_true = Y_test
y_predi = Y_pred
#print(pd.crosstab(y_true, y_predi, rownames=['True'], colnames=['Predicted'], margins=True))
print(classification_report(y_true, y_predi))
print(accuracy_score(y_true, y_predi))


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
