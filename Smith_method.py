#coding=utf-8

from utils.structure import structure
from utils.learning_rate_finder import LearningRateFinder
import pickle
import numpy as np
import matplotlib.pyplot as plt

model = structure(3, 950, 'mean_squared_error', 'relu', 'random_uniform', 'adam', 0)

lrf = LearningRateFinder(model)

with open('logarithmic/sample.pickle', 'rb') as handle:
	x = pickle.load(handle)
with open('logarithmic/sample_out.pickle', 'rb') as handle:
    y = pickle.load(handle)

lrf.find(
    x, y,
    1e-10, 1e+1,
    stepsPerEpoch=np.ceil((len(x) / float(1024))),
    batchSize=1024)

lrf.plot_loss()
#Método de smith
plt.savefig('Learning_Rate')