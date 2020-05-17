#coding=utf-8

from utils.structure import structure
from utils.learning_rate_finder import LearningRateFinder
from utils.sample import Sample
import pickle
import numpy as np
import matplotlib.pyplot as plt

model = structure(3, 950, 'mean_squared_error', 'relu', 'random_uniform', 'adam', 0)

lrf = LearningRateFinder(model)

muestra = Sample(ratio=[0.4, 1.6], T=[0.2, 1.1], r=[0.02, 0.1], o=[0.01, 1.0])

muestra.create('sample', 10**4, log=True)

x, y = muestra.open('sample', log=True)

lrf.find(
    x, y,
    1e-10, 1e+1,
    stepsPerEpoch=np.ceil((len(x) / float(1024))),
    batchSize=1024)

lrf.plot_loss()
#Método de smith
plt.savefig('Learning_Rate')
