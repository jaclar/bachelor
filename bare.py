import ceuler as eu
import numpy as np
import pylab as pl

def lan(a,t,delta,g,gamma,xi,f):
    return 1.0j*delta*a \
        - gamma*a \
        + f(t)  \
        - 1.0j*g*np.abs(a)**2*a 
    
def noise(delta,g,gamma,xi,f):
    return np.sqrt(gamma)*xi()


t = np.arange(0,100.0,0.01)
btw = 10
y0 = [16.57-23.3j,0.0,16.58+23.3j]

delta = 7.44
g = 0.009
gamma = 0.012

xi = lambda: np.random.normal(0.0,1.0)
f = lambda t: 0.6


for y in y0:
    for i in range(1):
        print lan(y,0.0,delta,g,gamma,xi,f)
        k = eu.euler(lan,y,t,btw,noise, \
                         args=(delta,g,gamma,xi,f))

        pl.plot(t,np.real(k))
        pl.plot(t,np.imag(k))

pl.show()
