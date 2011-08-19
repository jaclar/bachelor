import scipy as sp
import scipy.integrate as spi

import numpy as np
import pylab as pl

def euler_step(f,h,x,t,D,xi1,xi2,args=()):
    t = np.array(t)
    x = np.array(x)
    k1 = h*f(x,t,*args)
    st = np.sqrt(h)*np.array([np.real(D[0,0]*xi1 + D[0,1]*xi2),
                              np.imag(D[0,0]*xi1 + D[0,1]*xi2),
                              np.real(D[1,0]*xi1 + D[1,1]*xi2),
                              np.imag(D[1,0]*xi1 + D[1,1]*xi2)])
    return x + k1 + st

def euler(f,x0,t,btw,D,xi1,xi2,args=()):
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
            x[i//btw] = euler_step(f,h,x[(i-1)//btw],t[(i-1)//btw],D,xi1[i],xi2[i],args=args)
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
