import scipy as sp
import scipy.integrate as spi

import numpy as np
import pylab as pl

def RK4_step(f,h,x,t,noise,args=()):
    t = np.array(t)
    x = np.array(x)
    k1 = h*f(x,t,*args)
    k2 = h*f(x+1.0/2*k1, t+1.0/2*h, *args)
    k3 = h*f(x+1.0/2*k2, t+1.0/2*h, *args)
    k4 = h*f(x+k3, t+h, *args)
    st = np.sqrt(h)*noise(*args)
    return x + 1.0/6*k1 + 1.0/3*k2 + 1.0/3*k3 + 1.0/6*k4 + st

def RK4(f,x0,t,btw,noise,args=()):
    steps = len(t)*btw
    h = (t[1]-t[0])/btw

    if np.isscalar(x0):
        lx = 1
    else:
        lx = len(x0)
    x = np.zeros((len(t),lx))
    x[0] = x0

    np.seterr(all='raise')

    try:
        for i in range(1,steps):
            x[i//btw] = RK4_step(f,h,x[(i-1)//btw],t[(i-1)//btw],noise,args=args)
            # print i//btw, (i-1)//btw, x[i//btw]
    except FloatingPointError:
        print "diverged :("
    return x


# def f(x,t):
#     return x

# t = np.arange(0.0,20.0,0.01)

# o = spi.odeint(f,1.0,t)
# x = RK4(f,1.0,t,btw=10)

# print x

# pl.plot(x,label="kutta")
# #pl.plot(e,label="kutta-exp-theo")
# pl.plot(o,label="odeint")
# #pl.plot(np.exp(t),label="odeint-theo")
# pl.legend()
# pl.show()
