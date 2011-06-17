import numpy as np
import pylab as pl
import kutta as ku
import save
import multiprocessing as mp
import Queue

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


def integrate(lan, y0, t, delta, g, gamma, xi, f, n):
    print "run the worker"
    print y0, delta, g, gamma
    data = [t]
    for i in range(n):
        k = ku.RK4(lan,y0,t, \
                       args=(delta,g,gamma,xi,f))
        data.append(k)
    save.save(xi_var,xi_mean,gamma,delta,g,data)
    print "calling the worker finished"
        

xi_var = 0.001
xi_mean = 0.0
xi = lambda: np.random.normal(xi_mean,xi_var)

f = lambda t: 2.0

gamma = 0.05
delta = 0.03

t = np.arange(0,500.0,0.05)
y0 = [0.1,0.2]

g_array = np.arange(0.0,2.0,0.2)

n = 5

pool = mp.Pool(processes=4)
result = pool.apply_async(integrate,[(lan,y0,t,g,gamma,xi,f,n) for g in g_array])

