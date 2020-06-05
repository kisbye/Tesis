#coding=utf-8
from keras.callbacks.callbacks import LearningRateScheduler
import numpy as np
from utils.structure import k_fold

LAYERS = range(1,11)
NEURONS = range(50,1001,50)
ACTIVATIONS = ['elu', 'relu']
INITIALIZATIONS = ['he_uniform', 'glorot_uniform', 'random_uniform']
OPTIMIZERS = [ 'sgd','adam', 'rmsprop']
LOSSES = ['mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error']
DROPOUTS = np.linspace(0,0.2,5)
BATCHES = range(256,1409,128)

#k es el k del k_fold_cross_validation
#x son los datos de entrada de la red
#y son los output
class HyperParameter:

    def __init__(self, k, x, y, layers=1, neurons=50, activation='relu',
                 initialization='glorot_uniform', optimizer='sgd',
                 loss='mean_squared_error', dropout=0, batch=1024, epoch=200):
        
        self.k = k
        self.x = x
        self.y = y
        self.layers = layers
        self.neurons = neurons
        self.activation = activation
        self.initialization = initialization
        self.optimizer = optimizer
        self.loss = loss
        self.dropout = dropout
        self.batch = batch
        self.epoch = epoch

    
    def search_structure(self, layers=LAYERS, neurons=NEURONS):
        
        best_metric = 100
        struct = 0, 0
        for layrs in layers:
            for neurns in neurons:
                print('\n\nCapas, Neuronas:  ',layrs, neurns)
                
                metric = k_fold(self.k, self.x, self.y, layrs, neurns,
                                self.loss, self.activation,
                                self.initialization, self.optimizer,
                                self.batch, self.dropout, self.epoch)
                
                if metric < best_metric:
                    struct = layrs, neurns
                    best_metric = metric

        self.layers, self.neurons = struct

        return struct, best_metric


    def search_parameters(self, activations=ACTIVATIONS,
                        initializations=INITIALIZATIONS, optimizers=OPTIMIZERS):

        best_metric = 100
        parameters = '', '', ''

        for activation in activations:
            for initialization in initializations:
                for optimizer in optimizers:
                    print('\nActivation: {} ,Initialization: {}, Optimizer: {}'
                        .format(activation, initialization, optimizer))
                    
                    metric = k_fold(self.k, self.x, self.y, self.layers,
                                    self.neurons, self.loss, activation,
                                    initialization, optimizer, self.batch,
                                    self.dropout, self.epoch)

                    if metric < best_metric:
                        parameters = activation, initialization, optimizer
                        best_metric = metric

        self.activation, self.initialization, self.optimizer = parameters

        return parameters

    def search_loss_function(self, losses=LOSSES):
        
        best_metric = 100
        best_loss = 'mean_squared_error'
        
        for loss in losses:
            print('\nloss:  ',loss)

            metric = k_fold(self.k, self.x, self.y, self.layers,
                            self.neurons, loss, self.activation,
                            self.initialization, self.optimizer,
                            self.batch, self.dropout, self.epoch)

            if metric < best_metric:
                
                best_loss = loss
                best_metric = metric

        self.loss = best_loss

        return best_loss

    def search_dropout(self, dropouts=DROPOUTS):
        best_metric = 100
        best_dropout = 0 

        for dropout in dropouts:
            print('\ndropout:  ',dropout)
            
            metric = k_fold(self.k, self.x, self.y, self.layers,
                            self.neurons, self.loss, self.activation,
                            self.initialization, self.optimizer,
                            self.batch, dropout, self.epoch)

            if metric < best_metric:
                best_dropout = dropout
                best_metric = metric
        
        self.dropout = best_dropout

        return best_dropout

    def search_batch(self, batchs=BATCHES):
        best_batch = 256
        best_metric = 100

        

        for batch in batchs:
            print('\n\nBatch:  ',batch)
            
            metric = k_fold(self.k, self.x, self.y, self.layers,
                            self.neurons, self.loss, self.activation,
                            self.initialization, self.optimizer,
                            batch, self.dropout, self.epoch)

            if metric < best_metric:
                best_batch = batch
                best_metric = metric

        self.batch = best_batch

        return best_batch
            
    def get_parameters(self):
        string ='layers: {}\nneurons: {}\nactivation: {}\ninitialization: {}\noptimizer: {}\nloss: {}\nbatch: {}\ndroptout: {}'
        print(string.format(self.layers, self.neurons, self.activation,
                            self.initialization, self.optimizer, self.loss,
                            self.batch, self.dropout))
        
