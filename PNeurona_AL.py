
import pybrain  
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

import pandas as pd
import numpy as np
import csv

df = pd.read_csv("winecsv.csv")
print(df)

net = buildNetwork(4,2,1)

ds = SupervisedDataSet(4, 1)
ds.setField('input',df[['fx.acidity','vol.acidity','citacid','totsuldiox']])
ds.setField('target',df[['quality']])


trainer = BackpropTrainer(net, ds)
for i in range(15):
    trainer.train()

r1 = net.activate([12,.6,.6,22]) #bajo
r2 = net.activate([5,.3,.3,25]) #

print(r1),
print(r2),
