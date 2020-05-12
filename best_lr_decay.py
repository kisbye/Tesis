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
import tensorflow as tf

GPUs = len(tf.config.experimental.list_physical_devices('GPU'))
CPUs = len(tf.config.experimental.list_physical_devices('CPU'))

if GPUs > 0:
  
  print("Num GPUs Available: ", GPUs)
  print("Num CPUs Available: ", CPUs)
  config = tf.compat.v1.ConfigProto( device_count = {'GPU': GPUs , 'CPU': CPUs} ) 
  sess = tf.compat.v1.Session(config=config) 
  tf.compat.v1.keras.backend.set_session(sess)

muestra = Sample([0.4, 1.6], [0.2, 1.1], [0.02, 0.1], [0.01, 1.0])

muestra.create('sample', 10**4, log=True)

x, y = muestra.open('sample', log=True)

model = structure(3, 950, 'mean_squared_error', 'relu', 'random_uniform', 'adam', 0)

# Las variables varian proximas al area de aprendizaje obtenidas del 
# método de de Smith

expon = ExponentialDecay(model, x, y, [0.0005, 0.00075, 0.001, 0.0025, 0.005, 0.0075, 0.01], [0.007, 0.006, 0.005, 0.004, 0.003, 0.002])

expon.grid_search()


timety = TimeDecay(model, x, y, [0.0005, 0.00075, 0.001, 0.0025, 0.005, 0.0075, 0.01], [0.8, 0.825, 0.85, 0.875, 0.9, 0.95])

timety.grid_search()

stepy =StepDecay(model, x, y, [0.0005, 0.00075, 0.001, 0.0025, 0.005, 0.0075, 0.01], [0.8, 0.825, 0.85, 0.875, 0.9, 0.95], [5,10,15,20,30,40,50])

stepy.grid_search()

print(expon)
print(timety)
print(stepy)