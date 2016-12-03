
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
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from math import sqrt
from sklearn import metrics

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

clf = SVC(C=15, kernel='sigmoid', gamma=1, probability=True)
clf = clf.fit(Xscaler,np.ravel(Y_train))
print(len(Xscaler),len(Y_train))

#Definimos Y predecida
Y_pred=clf.predict_proba(XscalerTe) #decision_function


Y_pred2 = []
for i in range(len(Y_pred)):
    yp = pd.DataFrame(Y_pred[i]).max()
    Y_pred2.append(yp)
print(pd.DataFrame(Y_pred).head())
print(pd.DataFrame(Y_pred2).head())


# Graficamos la curva ROC
fpr, tpr, thresholds = roc_curve(Y_test, Y_pred2) #AQUI
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(fpr, tpr, lw=2, label='Roc CurveSVM', color='gray')
plt.axis([-0.5, 1.05, -0.05, 1.05])
#for x in range(0,len(Y_test),1):
#    a = thresholds[x]
#    b = [fpr[x],tpr[x]]
#    ax.annotate('(%s)' % a,xy=b,size='small')
plt.title("Roc CurveSVM", color='darkblue')
plt.plot([-.05, 1.5], [-.05, 1.5], '--', color=(.6, .6, .6), label='Barrera')
plt.xlabel('FP', color='red')
plt.ylabel('TP', color="red")
plt.legend(loc="upper left")
plt.show()
print(metrics.auc(fpr, tpr))


dist=  map(sqrt,(1-tpr)**2+(fpr**2))
ind = dist.index(min(dist))
print(thresholds[ind])

print(pd.DataFrame(Y_pred2).tail()) #AQUI
Y_pred3 = []
umbral = thresholds[ind]

for i in range(len(Y_pred)):
    if Y_pred2[i].any >=umbral: #AQUI
        r_cat = 1
    else:
        r_cat = 0
    Y_pred3.append(r_cat) #AQUI

print(pd.DataFrame(Y_pred3).tail()) #AQUI



#Hacemos la matriz de confusion
y = np.array(Y_test)
class_names = np.unique(y)
print(class_names)

cmtrx = confusion_matrix(Y_test,Y_pred3,labels=class_names) #AQUI
labels = class_names
print("Confusion matrix:\n%s" % cmtrx)

#Metrics
MSE = mean_squared_error(Y_test, Y_pred3)  #AQUI
print('MSE SVM',MSE)
y_true = Y_test
y_predi = Y_pred3 #AQUI
#print(pd.crosstab(y_true, y_predi, rownames=['True'], colnames=['Predicted'], margins=True))
print('classf_report',classification_report(y_true, y_predi))
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
plt.title('Confusion Matrix')
plt.xticks(range(width), labels[:width])
plt.yticks(range(height), labels[:height])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('confusion_matrix.png', format='png')
plt.show()


