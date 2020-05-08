#coding=utf-8
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras import losses
import gc
from keras.callbacks.callbacks import LearningRateScheduler, EarlyStopping


def structure(layers, neurons, loss, activation, initialization, optimizer, dropout):
    model = Sequential()
    model.add(Dropout(dropout, input_shape=(4,)))
    model.add(Dense(neurons, kernel_initializer=initialization, input_dim=4, activation= activation))

    for i in range(layers-1):
        model.add(Dropout(dropout, input_shape=(neurons,)))
        model.add(Dense(neurons, kernel_initializer=initialization,  activation= activation))

    model.add(Dense(1))

    model.compile(loss=loss, optimizer=optimizer, metrics=['mse', 'mae' ,'mape'])

    return model



def partic(muest, j, k):
    if j == 0:
        val = muest[:int(0.5+len(muest)/k)]
        result = muest[1*int(0.5+len(muest)/k):2*int(0.5+len(muest)/k)]
        st = 2
    else:
        result = muest[:int(0.5+len(muest)/k)]
        st = 1
        
    for i in range(st,k):
        if j == i:
            if j == k-1:
                return result, muest[i*int(0.5+len(muest)/k):]
            val = muest[i*int(0.5+len(muest)/k):(i+1)*int(0.5+len(muest)/k)]
            
        elif i == k-1:
            result = np.concatenate((result, muest[i*int(0.5+len(muest)/k):]))
        else:
            result = np.concatenate((result, muest[i*int(0.5+len(muest)/k):(i+1)*int(0.5+len(muest)/k)]))
    return result, val
    
def my_split(x_muest, y_muest, i, k):
    x_fold, x_validation = partic(x_muest, i, k)
    y_fold, y_validation = partic(y_muest, i, k)
    return x_fold, y_fold, x_validation, y_validation

def k_fold(k, x_muest, y_muest, layers, neurons, loss, activation, initialization, optimizer, batch, dropout, epoch):

    result = 0
    #shuffle
    x_cpy = x_muest.copy()
    y_cpy = y_muest.copy()
    indices = np.arange(len(x_muest))
    np.random.shuffle(indices)
    x_cpy = x_cpy[indices]
    y_cpy = y_cpy[indices]
    # learning rate
    
    
    for i in range(k):
        x_train, y_train, x_val, y_val = my_split(x_cpy, y_cpy, i, k)

        lrate = LearningRateScheduler(lambda x : 0.001,1)

        model = structure(layers, neurons, loss, activation, initialization, optimizer, dropout)
        
        history = model.fit(x_train, y_train,
                    epochs=epoch,
                    verbose = 0,
                    callbacks=[lrate],
                    batch_size=batch)
        
        result += model.evaluate(x_val, y_val)[1]
        del model
        gc.collect()
        
    return result