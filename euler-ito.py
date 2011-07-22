import scipy as sp
import scipy.integrate as spi
import kutta as ku

import numpy as np
import pylab as pl

def EI_step(f,h,x,t,args=()):
    t = np.array(t)
    x = np.array(x)
    stoch = np.random.normal(0.0,1.0)*np.sqrt(h)
    return x + f(x,t,*args)*stoch

def EI(f,x0,t,args=()):
    lt = len(t)
    if np.isscalar(x0):
        lx = 1
    else:
        lx = len(x0)
    x = np.zeros((lt,lx))
    x[0] = x0
    for i in range(1,lt):
        h = t[i]-t[i-1]
        x[i] = EI_step(f,h,x[i-1],t[i-1],args=args)
    return x

def sde(x,t,*args):
    return 0.0

N = 40
x = np.zeros(N+1)
e = np.zeros(N+1)
h = 0.01

x[0]=0.0
e[0]=0.0

t = np.arange(0.0,20.0,h)


def noise(delta,g,gamma,xi,f):
    return np.sqrt(gamma)*xi()

delta = 0.0
gamma = 1.0
g = 0.0
xi = lambda: np.random.normal(0,1.0)
f = lambda t: 1.0
btw = 1


for h in np.arange(0.001,0.1,0.02):
    t = np.arange(0.0,20.0,h)

    # xm = spi.odeint(f,1.0,t)

    mean = np.zeros((len(t),1))
    var = np.zeros((len(t),1))
    for i in range(100):
        x = ku.RK4(sde,0.0,t,btw,noise, args=(delta,g,gamma,xi,f))
        var += x**2
        mean += x
    var = var/100.
    mean = mean/100.
    print h, ":", np.sum(var)/len(t)
    pl.plot(t,var,label="euler: %f"%h)

pl.legend(loc=2)
pl.show()
