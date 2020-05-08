#coding=utf-8
from utils.structure import structure
from utils.sample import Sample
from keras.callbacks.callbacks import LearningRateScheduler
import pickle as plk
import math
import keras
import tensorflow as tf

GPUs = len(tf.config.experimental.list_physical_devices('GPU'))
CPUs = len(tf.config.experimental.list_physical_devices('CPU'))

if GPUs > 0:
  
  print("Num GPUs Available: ", GPUs)
  print("Num CPUs Available: ", CPUs)
  config = tf.compat.v1.ConfigProto( device_count = {'GPU': GPUs , 'CPU': CPUs} ) 
  sess = tf.compat.v1.Session(config=config) 
  tf.compat.v1.keras.backend.set_session(sess)


model = structure(3, 950, 'mean_squared_error', 'relu', 'random_uniform', 'adam', 0)

muestra = Sample(ratio=[0.4, 1.6], T=[0.2, 1.1], r=[0.02, 0.1], o=[0.01, 1.0])
          
muestra.create('train', 10**6, log=True)

x_train, y_train = muestra.open('train', log=True)

muestra.create('test', 10**6, log=True)

x_test, y_test = muestra.open('test', log=True)

muestra.create('validation', 10**5, log=True)

x_val, y_val = muestra.open('validation', log=True)


def step_decay(epoch):
    lrate = 0.001 * math.pow(0.9,  
           math.floor((1+epoch)/10))
    return lrate


def my_stepy(epoch):
	lrate = 0.0005 * math.pow(0.9,  
           math.floor((1+epoch)/10))
	if epoch > 500:
		return 50*lrate
	return lrate

#obtengo un mejor resultado
def my_stepy_2(epoch):
  i = 0

  lrate = 0.0005 * math.pow(0.9,  
           math.floor((1+epoch)/10))
  while lrate < 10**-(6+i):
    lrate *= 100
    i  += 0.5
  return lrate


lrate = LearningRateScheduler(my_stepy_2,1)

history_stepd = model.fit(x_train, y_train,
                    epochs=3000,
                    batch_size=1024,
                    callbacks=[lrate],
                    validation_data=(x_val, y_val))

    #mse error cuadratico medio
print(model.evaluate(x_test, y_test))
model.save("second.h5")
