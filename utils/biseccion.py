#coding=utf-8

def bisec(f, a, b, tol):
	if f(a) == 0:
		return a
	if f(b) == 0:
		return b

	if f(a)*f(b) > 0:
		return -1
	assert f(a)*f(b) < 0, "No cumple pre-condición"
	o = 0 #volatilidad
	while(tol<abs(b-a)):
		o = (a+b)/2
		tmp = f(o)
		if tmp > 0:
			b = o
		elif tmp < 0:
			a = o
		else:
			return o
	return o

