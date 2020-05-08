#coding=utf-8

import pickle
import gc
from keras.callbacks.callbacks import LearningRateScheduler
import numpy as np
from utils.structure import my_split,structure
from utils.sample import Sample
from keras.models import clone_model
from utils.decay_lr import ExponentialDecay, TimeDecay, StepDecay
import math


muestra = Sample([0.4, 1.6], [0.2, 1.1], [0.02, 0.1], [0.01, 1.0])

muestra.create('sample', 10**4, log=True)

x, y = muestra.open('sample', log=True)

model = structure(3, 950, 'mean_squared_error', 'relu', 'random_uniform', 'adam', 0)

# Las variables varian proximas al area de aprendizaje obtenidas del 
# método de de Smith

expon = ExponentialDecay(model, x, y, [0.0005, 0.00075, 0.001, 0.0025, 0.005, 0.0075, 0.01], [0.0025, 0.005, 0.009, 0.0095, 0.01, 0.015])

expon.grid_search()

timety = TimeDecay(model, x, y, [0.0005, 0.00075, 0.001, 0.0025, 0.005, 0.0075, 0.01], [0.01, 0.05, 0.1, 0.5, 1, 1.5, 2, 4, 8])

timety.grid_search()

stepy =StepDecay(model, x, y, [0.0005, 0.00075, 0.001, 0.0025, 0.005, 0.0075, 0.01], [0.8, 0.825, 0.85, 0.875, 0.9, 0.95], [5,10,15,20,30,40,50])

stepy.grid_search()

print(stepy)