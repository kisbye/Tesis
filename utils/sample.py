#coding=utf-8
from utils.black_scholes import call_price_ratio
import numpy as np
import pickle
import math
from os import path, mkdir, strerror
from errno import ENOENT


# c: valor de la opción call
# k: Strike
# s0: Precio del subyacente en T=0
# r: tasa libre de riesgo
# T: tiempo de maduración
# o: volatilidad
# R: Ratio(S(t)/k)

class Sample:
    def __init__(self, ratio=None, T=None, r=None, o=None):
        self.ratio = ratio
        self.T = T
        self.r = r
        self.o = o
        

    def create(self, filename, N, log=None):
        if not N:
            N = self.N

        if log:
            return self.create_logarithmic(filename, N)
        else:
            return self.create_standar(filename, N)
        
    def create_standar(self, filename, N):
        result = []
        sampl = []
        
        i = 0
    
        while i < N:
            ratio = np.random.uniform(self.ratio[0], self.ratio[1])
            T = np.random.uniform(self.T[0], self.T[1])
            r = np.random.uniform(self.r[0], self.r[1])
            o = np.random.uniform(self.o[0], self.o[1])
            c = call_price_ratio(ratio, r, o, T)
            if c > 0:
                result.append(o)
                sampl.append([c , ratio, r, T])
                
                i += 1
        if not path.exists('standar'):
            mkdir('standar')
        with open('standar/{}.pickle'.format(filename), 'wb') as handle:
            pickle.dump(np.array(sampl), handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('standar/{}_out.pickle'.format(filename), 'wb') as handle:
            pickle.dump(np.array(result), handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('{} is done ...'.format(filename))
        

    def create_logarithmic(self, filename, N):
        sampl = []
        result = []
        i = 0
    
        while i < N:
            ratio = np.random.uniform(self.ratio[0], self.ratio[1])
            T = np.random.uniform(self.T[0], self.T[1])
            r = np.random.uniform(self.r[0], self.r[1])
            o = np.random.uniform(self.o[0], self.o[1])
            c = call_price_ratio(ratio, r, o, T)
            V_tilde = c - max(ratio - math.exp(-r*T), 0)

            if V_tilde <= 0:
                continue

            aux = math.log(V_tilde)

            if not(-16.12 <= aux <= -0.94):
                continue
            result.append(o)
            sampl.append([aux , ratio, r, T])
        
            i += 1
        if not path.exists('logarithmic'):
            mkdir('logarithmic')
        with open('logarithmic/{}.pickle'.format(filename), 'wb') as handle:
            pickle.dump(np.array(sampl), handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('logarithmic/{}_out.pickle'.format(filename), 'wb') as handle:
            pickle.dump(np.array(result), handle, protocol=pickle.HIGHEST_PROTOCOL)

        print('{} is done ...'.format(filename))
        

    def open(self, filename, log = None):
        directory = 'standar'
        if log:
            directory = 'logarithmic'

        try:
            with open('{}/{}.pickle'.format(directory, filename), 'rb') as handle:
                x = pickle.load(handle)
            with open('{}/{}_out.pickle'.format(directory, filename), 'rb') as handle:
                y = pickle.load(handle)
        except FileNotFoundError:
            print(FileNotFoundError(ENOENT, strerror(ENOENT), filename))
            return [], []    
        return x, y


