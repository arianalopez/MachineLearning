import random
import sys
import os
import winpython
import pip
import csv
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
# matplotlib inline
import numpy as np

#autor Ariana Lopez
''' EJERCICIO 1: Genere una lista de numeros aletorios del 0 al 100
 Haga otra lista donde categorice los valores de la primera como
 alto medio y bajo'''


RandNum = [random.randrange(1, 101, 1) for i in range(100)]
# print(RandNum,' ',end='')
NewList = []

for i in RandNum:
    if (i <= 33):
        b = ['bajo', i]
        NewList.append(b)
    elif (i <= 66):
        m = ['medio', i]
        NewList.append(m)
    elif (i <= 100):
        a = ['alto', i]
        NewList.append(a)
    else:
        break

for n in NewList:
    print(n)

''' EJERCICIO 2: Haga una funcion que dada una lista, regrese la lista invertida '''

print('\n'*3)

MyList= [2,4,7,8,3]
InvList = []

i = 0
while (i < len(MyList)):
    y = int(len(MyList)- i)-1
    z = (MyList[y])
    InvList.append(z)
    i += 1
print(InvList)




''' EJERCICIO 3: Convierta dos listas de la misma longitud, en un
diccionario en donde una tiene las llaves y otra los valores '''
print('\n'*3)

#diccionario en dos listas
key_superheroe = ['Batman','Superman','Flash','WonderWoman']
value_identidad = ['Bruce Wayne','Clark Kent','Bally Allen','Diana Prince']
dictionary = dict(zip(key_superheroe,value_identidad))
print(dictionary)

#diccionario en una lista
identidad_superheroe = {'Batman': 'Bruce Wayne',
                'Superman':'Clark Kent',
                'Flash':'Barry Allen',
                'WonderWoman':'Diana Prince'}
print(identidad_superheroe.keys())
print(identidad_superheroe.values())