
import pybrain
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

import pandas as pd
import numpy as np
import csv

df = pd.read_csv("datatest.csv")
print(df)

net = buildNetwork(3,12,1)

ds = SupervisedDataSet(3, 1)
ds.setField('input',df[['Age','Income','Saldo']])
ds.setField('target',df[['Riesgo']])


trainer = BackpropTrainer(net, ds)
for i in range(900):
    trainer.train()

r1 = net.activate([39,43674,41038])
r2 = net.activate([20,7000,8000])

print(r1)
print(r2)

