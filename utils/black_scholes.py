#coding=utf-8

import numpy as np
import math
import scipy.stats as st


# c: valor de la opción call
# k: Strike
# s0: Precio del subyacente en T=0
# r: tasa libre de riesgo
# T: tiempo de maduración


# auxiliar de call_price
def d1(S0,k, r, o, T):
	ln = np.log(S0/float(k))
	mu = (r + (o**2)/float(2))*T
	return (ln + mu)/float((o*math.sqrt(T)))


def call_price(S0, k, r, o, T):
	d_1 = d1(S0, k, r, o, T)
	d_2 = d_1 - o*math.sqrt(T) 
	norm1 = st.norm.cdf(d_1)
	norm2 = st.norm.cdf(d_2)
	return S0*norm1 - k*math.exp(-r*T)*norm2

def raiz(c, S0, k, r, o, T):
	return call_price(S0, k, r, o, T) - c

#d1 auxiliar de la fórmula de Black-Scholes
def d1_ratio(ratio, r, o, T):
	ln = np.log(ratio)
	mu = (r + (o**2)/float(2))*T
	return (ln + mu)/float(o*math.sqrt(T))

#Fórmula de Black-Scholes
def call_price_ratio(ratio, r, o, T):
	d_1 = d1_ratio(ratio, r, o, T)
	d_2 = d_1 - o*math.sqrt(T)
	norm1 = st.norm.cdf(d_1)
	norm2 = st.norm.cdf(d_2)
	
	return ratio*norm1 - math.exp(-r*T)*norm2

#función para aplicar método de bisección o de brent
def raiz_ratio(c, ratio, r, o, T):
	return call_price_ratio(ratio, r, o, T) - c

