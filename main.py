import numpy as np
import pylab as pl
import kutta as ku
import scipy.integrate as integrate
import save

def lan(a,t,delta,g,gamma,xi,f):
    a = a[0] + 1.0j*a[1]
    r = 1.0j*delta*a \
        - gamma*a \
        + f(t)*(1.0 + 1.0j) \
        + g*np.abs(a)**2*a 
        # + np.sqrt(2*gamma)*xi()*(1.0+1.0j)
    return np.array([r.real,r.imag],dtype=np.float64)

def noise(delta,g,gamma,xi,f):
    return np.sqrt(2*gamma)*xi()

def run(y0, t, btw, delta, g, gamma, f, n = 1):
    print y0, delta, g, gamma
    data = [t]
    xi_mean = 0.0
    xi_var = 1.0
    xi = lambda: np.random.normal(xi_mean,xi_var)

    y0 = np.array(y0)
    if y0.shape == (2,):
        y0 = np.array([y0])
    for y in y0:
        for i in range(n):
            print y
            k = ku.RK4(lan,y,t,btw,noise, \
                          args=(delta,g,gamma,xi,f))
            # k = integrate.odeint(lan,y,t, \
            #                        args=(delta,g,gamma,xi,f))
            data.append(k)
    save.save(xi_var,xi_mean,gamma,delta,g,f,data,"data")
        
# f_array = [lambda t: p for p in np.arange(0.0,1.0,0.3)]
f_array = [lambda t: 0.3,lambda t: 1.0, lambda t: 3.0]
f_array = [lambda t: 10.0]

gamma_array = np.arange(0.001,0.1,0.02)
gamma_array = [0.5]

# delta_array = np.arange(0.0,1.0,0.3)
delta_array = [1.0]

t = np.arange(0,80.0,0.005)
btw = 10
y0 = np.array([-0.5,0.9],dtype=np.float64)

# g_array = np.arange(0.0,1.0,0.3)
g_array = [-0.001]
y0 = []

for x in np.arange(-1.0,1.0,0.5):
    for y in np.arange(-1.0,1.0,0.1):
        y0.append((x,y))

y0 = [300.0,400.8]
n = 4


for g in g_array:
    for f in f_array:
        for gamma in gamma_array:
            for delta in delta_array:
                run(y0,t,btw,delta,g,gamma,f,n)
    

