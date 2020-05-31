#coding=utf-8
from keras.models import load_model
from utils.sample import Sample
from utils.black_scholes import raiz_ratio, d1_ratio, call_price_ratio
from utils.biseccion import bisec
import time
import pickle
import math
import numpy as np
from os import path, mkdir, strerror
from scipy.optimize import brentq
import tensorflow as tf



opn = Sample(ratio=[0.4, 1.6], T=[0.2, 1.1], r=[0.02, 0.1], o=[0.01, 1])

opn.create('prueba', N=10**5)
x_test, y_test = opn.open('prueba')

start_time = time.time()

j = 0
fails = []
i = 0
for c, ratio, r, T in x_test:

    def f(x):
        return raiz_ratio(c, ratio, r, x, T) #x seria la variable para aplicar bisección
    
    
    o = bisec(f, 0.01, 1, 10**-4) #error absoluto medio

    #Casos en los que no cumple la precondición para aplicar bisección
    if o == -1:
        fails.append([c, ratio, r, T, y_test[i]])
        j += 1
    i += 1

#guardo los casos en que no pude aplicar bisección  
if not path.exists('casos_fallidos'):
    mkdir('casos_fallidos')
with open('casos_fallidos/fallidos.pickle', 'wb') as handle:
    pickle.dump(np.array(fails), handle, protocol=pickle.HIGHEST_PROTOCOL)


time_biseccion = time.time() - start_time

start_time = time.time()

for c, ratio, r, T in x_test:

    f = lambda x: raiz_ratio(c, ratio, r, x, T)#x seria la variable para aplicar brents
    # que se cumpla la precondicción
    if f(0.01) < 0:
        o = brentq(f, 0.01, 1, xtol=10**-4, maxiter=2000)#error absoluto medio
    


time_brent = time.time() - start_time


model = load_model('models/first.h5')

start_time = time.time()

mse, mae, mape = tuple(model.evaluate(x_test, y_test)[1:])

time_red = time.time() - start_time
   
opn.create('prueba', N=10**5, log=True)
x_test, y_test = opn.open('prueba', log=True)


model = load_model('models/second.h5')


mse_l, mae_l, mape_l = tuple(model.evaluate(x_test, y_test)[1:])



print('\n\nNo se pudo aplicar bisección en {} de {} casos'.format(j, i))

print('\n\nPrimera red')

print('error cuadratico medio: {}'.format(mse))

print('error absoluto medio: {}'.format(mae))

print('error porcentual medio: {}'.format(mape))

print('\n\nRed con logaritmo al ratio')

print('error cuadratico medio: {}'.format(mse_l))

print('error absoluto medio: {}'.format(mae_l))

print('error porcentual medio: {}'.format(mape_l))

print('\n\nTiempo de ejecucion red: {}'.format(time_red))

print('\n\nTiempo de ejecucion bisecion: {}'.format(time_biseccion))

print('\n\nTiempo de ejecucion brent: {}'.format(time_brent))
