#coding=utf-8
import pickle
import numpy as np
import matplotlib.pyplot as plt
from utils.structure import structure, k_fold
from utils.sample import Sample
import utils.k_fold_cross_validation as kf
import tensorflow as tf

GPUs = len(tf.config.experimental.list_physical_devices('GPU'))
CPUs = len(tf.config.experimental.list_physical_devices('CPU'))

#Uso de GPU
if GPUs > 0:
	
	print("Num GPUs Available: ", GPUs)
	print("Num CPUs Available: ", CPUs)
	config = tf.compat.v1.ConfigProto( device_count = {'GPU': GPUs , 'CPU': CPUs} ) 
	sess = tf.compat.v1.Session(config=config) 
	tf.compat.v1.keras.backend.set_session(sess)




#Creacion de muestra
muestra = Sample(ratio=[0.4, 1.6], T=[0.2, 1.1], r=[0.02, 0.1], o=[0.01, 1.0])

muestra.create('sample', 10**4, log=True)

x, y = muestra.open('sample', log=True)

best_per_layers = dict()

ob_hyp = kf.HyperParameter(8, x, y)


#Capas y neuronas
ob_hyp.search_structure()

# Función de activations, Optimizador de descenso del gradiente 
# e inicialización de Pesos
ob_hyp.search_parameters()

# Funcion de error
ob_hyp.search_loss_function()

# Dropout
ob_hyp.search_dropout()

# Tamaño de batch
ob_hyp.search_batch()

#imprimir los hiperparametros
ob_hyp.get_parameters()