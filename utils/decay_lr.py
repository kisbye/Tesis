#coding=utf-8

import pickle
import gc
from keras.callbacks.callbacks import LearningRateScheduler
import numpy as np
from utils.structure import my_split,structure
from utils.sample import Sample
from keras.models import clone_model
import math
#Exponential Decay

class DecayLearnigRate:

    def __init__(self, model, x, y, lrs, decays, epochs_drop=None):

        self.model = model
        self.x = x
        self.y = y
        self.lrs = lrs
        self.decays = decays
        self.epochs_drop = epochs_drop
        self.best_lr = 0
        self.best_decay = 0
        self.best_epoch_drop = 0
        self.best_loss = 100

    def __str__(self):
        return 'Learning Rate: {}, Decay: {}, epoch_drop: {}'.format(self.best_lr, self.best_decay, self.best_epoch_drop)

    def my_shuffle(self):
        x_cpy = self.x.copy()
        y_cpy = self.y.copy()

        indices = np.arange(len(self.x))
        np.random.shuffle(indices)

        x_cpy = x_cpy[indices]
        y_cpy = y_cpy[indices]

        return my_split(x_cpy, y_cpy, 0, 10)

class ExponentialDecay(DecayLearnigRate):
      
    def grid_search(self):

        for lr in self.lrs:
            for decay in self.decays:
                print('Learning Rate: {}\nDecay: {}'.format(lr, decay))
                copy_model = clone_model(self.model)

                copy_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mae' ,'mape'])

                x_train, y_train, x_val, y_val = self.my_shuffle()
                
                def exp_decay(epoch):
                    new_lr = lr*math.exp(-decay*epoch)
                    
                    if new_lr < 10**-9:
                        #learning rate muy chico, deja de entrenar
                        copy_model.stop_training = True

                    return new_lr
                
                lrate = LearningRateScheduler(exp_decay,1)

                history = copy_model.fit(x_train, y_train,
                                        epochs=1000,
                                        verbose = 0,
                                        callbacks=[lrate],
                                        batch_size=1024)
                
    
                result = copy_model.evaluate(x_val, y_val)[1]
                if self.best_loss > result:
                    self.best_loss = result
                    self.best_decay = decay
                    self.best_lr = lr

                del copy_model
                gc.collect()

class TimeDecay(DecayLearnigRate):
      
    def grid_search(self):

        for lr in self.lrs:
            for decay in self.decays:
                print('Learning Rate: {}\nDecay: {}'.format(lr, decay))
                copy_model = clone_model(self.model)

                copy_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mae' ,'mape'])
                
                x_train, y_train, x_val, y_val = self.my_shuffle()

                def time_decay(epoch):
                    new_lr = lr*1/(1+ decay*epoch)
                    
                    if new_lr < 10**-9:
                        # stop returning and return from the method
                        copy_model.stop_training = True

                    return new_lr
                
                lrate = LearningRateScheduler(time_decay,1)

                history = copy_model.fit(x_train, y_train,
                                        epochs=1000,
                                        verbose = 0,
                                        callbacks=[lrate],
                                        batch_size=1024)
                
    
                result = copy_model.evaluate(x_val, y_val)[1]
                if self.best_loss > result:
                    self.best_loss = result
                    self.best_decay = decay
                    self.best_lr = lr

                del copy_model
                gc.collect()

class StepDecay(DecayLearnigRate):
      
    def grid_search(self):

        for lr in self.lrs:
            for decay in self.decays:
                for epoch_drop in self.epochs_drop:
                    print('Learning Rate: {}\nDecay: {}\nEpoch Drop: {}'.format(lr, decay, epoch_drop))
                    copy_model = clone_model(self.model)

                    copy_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mae' ,'mape'])
                    
                    x_train, y_train, x_val, y_val = self.my_shuffle()

                    def step_decay(epoch):
                        new_lr = lr * math.pow(decay,  
                                             math.floor((1+epoch)/epoch_drop))
                        
                        if new_lr < 10**-8:
                            # stop returning and return from the method
                            copy_model.stop_training = True

                        return new_lr
                    
                    lrate = LearningRateScheduler(step_decay,1)

                    history = copy_model.fit(x_train, y_train,
                                            epochs=1000,
                                            verbose = 0,
                                            callbacks=[lrate],
                                            batch_size=1024)
                    
        
                    result = copy_model.evaluate(x_val, y_val)[1]
                    if self.best_loss > result:
                        
                        self.best_loss = result
                        self.best_decay = decay
                        self.best_lr = lr
                        self.best_epoch_drop = epoch_drop

                    del copy_model
                    gc.collect()
        print(self.best_lr, self.best_decay, self.best_epoch_drop)
