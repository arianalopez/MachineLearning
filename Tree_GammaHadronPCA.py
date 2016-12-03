
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
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA

XYdf = pd.read_csv("cherenkovGHray.csv")
#print(XYdf)

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


####WITH PCA
pca = PCA(n_components=0.9,whiten=True)
Xscaler = pca.fit_transform(Xscaler)
XscalerTe = pca.transform(XscalerTe)

Xscaler = pd.DataFrame(Xscaler)
XscalerTe = pd.DataFrame(XscalerTe)

#clf = tree.DecisionTreeClassifier()
clf = AdaBoostClassifier(tree.DecisionTreeClassifier(),n_estimators=5000,learning_rate=1)
clf = clf.fit(Xscaler,np.ravel(Y_train))


#Definimos Y predecida
Y_pred=clf.predict_proba(XscalerTe)
Y_pred = Y_pred[:,1]
print('TypeY_pred',type(Y_pred))
print(Y_pred)
#print(zip(clf.classes_, clf.predict_proba(XscalerTe)[0]))
#tree.export_graphviz(clf,out_file='tree.dot')
#dot_file = tree.export_graphviz(clf.tree_, out_file='tree_d1.dot')  #export the tree to .dot file


# Graficamos la curva ROC
fpr, tpr, thresholds = roc_curve(Y_test, Y_pred)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(fpr, tpr, lw=2, label='Roc Curve DT', color='darkblue')
plt.axis([-0.4, 1.05, -0.05, 1.05])
for x in range(0,len(thresholds),10):
    a = round(thresholds[x],2)
    b = [fpr[x],tpr[x]]
    ax.annotate('(%s)' % a,xy=b,size='small', color='gray')
plt.title("Roc Curve DT", color='darkred')
plt.plot([-.05, 1.5], [-.05, 1.5], '--', color=(.6, .6, .6), label='Barrera')
plt.xlabel('FP', color='black')
plt.ylabel('TP', color="black")
plt.legend(loc="upper left")
plt.show()
print('threshauc',metrics.auc(fpr, tpr))

#Con el umbral de la ROC CURVE hacemos la Matriz de confusion


dist=  map(sqrt,(1-tpr)**2+(fpr**2))
ind = dist.index(min(dist))
print('threshind',thresholds[ind])


Y_pred2 = []
umbral = thresholds[ind]

for i in range(len(Y_pred)):
    if Y_pred[i] >=umbral:
        r_cat = 1
    else:
        r_cat = 0
    Y_pred2.append(r_cat)

print(pd.DataFrame(Y_pred).tail())
print(pd.DataFrame(Y_pred2).tail())

print('typeYpred2',type(Y_pred2))

y = np.array(Y_test)
class_names = np.unique(y)
print('class names',class_names)

cmtrx = confusion_matrix(Y_test,Y_pred2,labels=class_names)
labels = class_names
print("Confusion matrix DT:\n%s" % cmtrx)

#Metrics
MSE = mean_squared_error(Y_test, Y_pred2)  #AQUI
print('MSE DT',MSE)
y_true = Y_test
y_predi = Y_pred2
#print(pd.crosstab(y_true, y_predi, rownames=['True'], colnames=['Predicted'], margins=True))
print(classification_report(y_true, y_predi))
print('accuracy_score',accuracy_score(y_true, y_predi))

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
plt.title('Confusion Matrix DT')
plt.xticks(range(width), labels[:width])
plt.yticks(range(height), labels[:height])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('confusion_matrix.png', format='png')
plt.show()
