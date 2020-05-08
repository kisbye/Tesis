#https://www.pyimagesearch.com/2019/08/05/keras-learning-rate-finder/
#coding=utf-8
from keras.callbacks import LambdaCallback
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import tempfile


class LearningRateFinder:
    def __init__(self, model, stopFactor=1, beta=0.98):
        
        self.model = model
        self.stopFactor = stopFactor
        self.beta = beta

        # initialize our list of learning rates and losses,
        # respectively
        self.lrs = []
        self.losses = []

        # initialize our learning rate multiplier, average loss, best
        # loss found thus far, current batch number, and weights file
        self.lrMult = 1
        self.avgLoss = 0
        self.bestLoss = 1e9
        self.batchNum = 0
        self.weightsFile = None
        
    def on_batch_end(self, batch, logs):
        # grab the current learning rate and add log it to the list of
        # learning rates that we've tried
        lr = K.get_value(self.model.optimizer.lr)
        self.lrs.append(lr)

        # grab the loss at the end of this batch, increment the total
        # number of batches processed, compute the average average
        # loss, smooth it, and update the losses list with the
        # smoothed value
        l = logs["loss"]
        self.batchNum += 1
        self.avgLoss = (self.beta * self.avgLoss) + ((1 - self.beta) * l)
        smooth = self.avgLoss / (1 - (self.beta ** self.batchNum))
        self.losses.append(smooth)

        # compute the maximum loss stopping factor value
        stopLoss = self.stopFactor 

        # check to see whether the loss has grown too large
        if self.batchNum > 1 and smooth > 1:
            # stop returning and return from the method
            self.model.stop_training = True
            return

        # check to see if the best loss should be updated
        if self.batchNum == 1 or smooth < self.bestLoss:
            self.bestLoss = smooth

        # increase the learning rate
        lr *= self.lrMult
        
        K.set_value(self.model.optimizer.lr, lr)
        
        
    def find(self, trainX, trainY, startLR, endLR, epochs=None,
        stepsPerEpoch=None, batchSize=1024, sampleSize=2048,
        verbose=1):
        # reset our class-specific variables
        #self.reset()


        # grab the number of samples in the training data and
        # then derive the number of steps per epoch
        numSamples = len(trainX)
        stepsPerEpoch = np.ceil(numSamples / float(batchSize))

        # if no number of training epochs are supplied, compute the
        # training epochs based on a default sample size
        if epochs is None:
            epochs = int(np.ceil(sampleSize / float(stepsPerEpoch)))

        # compute the total number of batch updates that will take
        # place while we are attempting to find a good starting
        # learning rate
        numBatchUpdates = epochs * stepsPerEpoch

        # derive the learning rate multiplier based on the ending
        # learning rate, starting learning rate, and total number of
        # batch updates
        self.lrMult = (endLR / startLR) ** (1.0 / numBatchUpdates)

        # create a temporary file path for the model weights and
        # then save the weights (so we can reset the weights when we
        # are done)
        self.weightsFile = tempfile.mkstemp()[1]
        self.model.save_weights(self.weightsFile)

        # grab the *original* learning rate (so we can reset it
        # later), and then set the *starting* learning rate
        origLR = K.get_value(self.model.optimizer.lr)
        K.set_value(self.model.optimizer.lr, startLR)

        # construct a callback that will be called at the end of each
        # batch, enabling us to increase our learning rate as training
        # progresses
        callback = LambdaCallback(on_batch_end=lambda batch, logs:
            self.on_batch_end(batch, logs))

    

        # otherwise, our entire training data is already in memory
        # train our model using Keras' fit method
        self.model.fit(
            trainX, trainY,
            batch_size=batchSize,
            epochs=epochs,
            callbacks=[callback],
            verbose=verbose)

        # restore the original model weights and learning rate
        self.model.load_weights(self.weightsFile)
        K.set_value(self.model.optimizer.lr, origLR)

    def plot_loss(self, skipBegin=10, skipEnd=1, title=""):
        # grab the learning rate and losses values to plot
        lrs = self.lrs[:-2]
        losses = self.losses[:-2]

        # plot the learning rate vs. loss
        plt.figure(num=None, figsize=(15, 12), dpi=80, facecolor='w', edgecolor='k')
        plt.plot(lrs, losses)
        
        plt.xscale("log")
        plt.xlabel("Learning Rate (Log Scale)", fontsize=18)
        plt.ylabel("Loss", fontsize=18)
        plt.xticks([10**-i for i in range(10, 0,-1)], fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid()

        # if the title is not empty, add it to the plot
        if title != "":
            plt.title(title)