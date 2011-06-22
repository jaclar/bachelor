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

def run(y0, t, delta, g, gamma, xi_var, f, n):
    print y0, delta, g, gamma
    data = [t]
    xi_mean = 0.0
    xi = lambda: np.random.normal(xi_mean,xi_var)

    for i in range(n):
        k = ku.RK4(lan,y0,t, \
                       args=(delta,g,gamma,xi,f))
        data.append(k)
    save.save(xi_var,xi_mean,gamma,delta,g,f,data,"data")
        

xi_var_array = np.arange(0.1,1.1,0.3)

f_array = [lambda t: p for p in np.arange(0.0,1.0,0.3)]

gamma_array = np.arange(0.0,1.0,0.3)
delta_array = np.arange(0.0,1.0,0.3)

t = np.arange(0,200.0,0.05)
y0 = [0.1,0.2]

g_array = np.arange(0.0,1.0,0.3)

n = 5


for g in g_array:
    for xi_var in xi_var_array:
        for f in f_array:
            for gamma in gamma_array:
                for delta in delta_array:
                    run(y0,t,delta,g,gamma,xi_var,f,n)
    

