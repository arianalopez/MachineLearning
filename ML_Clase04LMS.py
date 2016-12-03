import random
import sys
import os
import winpython
import pip
import sklearn
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import csv
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv("regLin.csv")
print (df)

#train_test_split: split arrays or matrices into random train and test subsets
#train_size: represent the proportion of the dataset to include in the train split
X_train, X_test, Y_train, Y_test = train_test_split(df[["X"]],df[["y"]], train_size=0.75)
print(X_train,Y_train)
