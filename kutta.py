import scipy as sp
import scipy.integrate as spi

import numpy as np
import pylab as pl

def RK4_step(f,h,x,t,args=()):
    t = np.array(t)
    x = np.array(x)
    k1 = h*f(x,t,*args)
    k2 = h*f(x+1.0/2*k1, t+1.0/2*h, *args)
    k3 = h*f(x+1.0/2*k2, t+1.0/2*h, *args)
    k4 = h*f(x+k3, t+h, *args)
    return x + 1.0/6*k1 + 1.0/3*k2 + 1.0/3*k3 + 1.0/6*k4

def RK4(f,x0,t,args=()):
    lt = len(t)
    if np.isscalar(x0):
        lx = 1
    else:
        lx = len(x0)
    x = np.zeros((lt,lx))
    x[0] = x0
    for i in range(1,lt):
        h = t[i]-t[i-1]
        x[i] = RK4_step(f,h,x[i-1],t[i-1],args=args)
    return x

# def f(x,t):
#     return x

# N = 40
# x = np.zeros(N+1)
# e = np.zeros(N+1)
# h = 0.1

# x[0]=1.0
# e[0]=1.0
# t = np.arange(0.0,2.0,0.1)

# # for i in range(N):
# #     x[i+1] = RK4_step(f,h,x[i],h*(i))
# #     print h*(i)
# #     e[i+1] = np.exp(h*(i+1))

# o = spi.odeint(f,1.0,t)
# x = RK4(f,1.0,t)

# print x
# print o
# pl.plot(x,label="kutta")
# #pl.plot(e,label="kutta-exp-theo")
# pl.plot(o,label="odeint")
# #pl.plot(np.exp(t),label="odeint-theo")
# pl.legend()
# pl.show()
