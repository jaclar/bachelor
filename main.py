import numpy as np
import pylab as pl
import kutta as ku
import save

def lan(a,t,delta,g,gamma,xi,f):
    a0 = -1*delta*a[1]              \
         + g*(a[0]**2+a[1]**2)*a[1] \
         - gamma*a[0]               \
         + f(t)                     \
         + xi()
    a1 =   delta*a[0]               \
         - g*(a[0]**2+a[1]**2)*a[0] \
         - gamma*a[1]               \
         + f(t)                     \
         + xi()
    return np.array([a0,a1])

xi_var = 0.001
xi_mean = 0.0
xi = lambda: np.random.normal(xi_mean,xi_var)
