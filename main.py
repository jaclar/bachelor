import numpy as np
import pylab as pl
import kutta as ku
import euler as eu
import scipy.integrate as integrate
import save
from multiprocessing import Process

def lan(a,t,delta,g,gamma,xi,f):
    a = a[0] + 1.0j*a[1]
    r = 1.0j*delta*a \
        - gamma*a \
        + f(t)  \
        - 1.0j*g*np.abs(a)**2*a 
        # + np.sqrt(2*gamma)*xi()*(1.0+1.0j)
    return np.array([r.real,r.imag],dtype=np.float64)

def noise(delta,g,gamma,xi,f):
    # return 0
    return np.sqrt(gamma)*xi()

def oup(a,t,theta,mu,sigma):
    return theta*(mu-a)

def oup_noise(theta,mu,sigma):
    return sigma*np.random.normal(0.0,1.0) 

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
            # k = ku.RK4(lan,y,t,btw,noise, \
            #              args=(delta,g,gamma,xi,f))
            # k = integrate.odeint(lan,y,t, \
            #                        args=(delta,g,gamma,xi,f))
            k = eu.euler(lan,y,t,btw,noise, \
                            args=(delta,g,gamma,xi,f))

            # k = eu.euler(oup,y,t,btw,oup_noise, \
            #                  args=(2.0,0.0,1.0))
            data.append(k)
    save.save(xi_var,xi_mean,gamma,delta,g,f,data,"data")
        
# f_array = [lambda t: p for p in np.arange(0.0,1.0,0.3)]
f_array = [lambda t: 0.3,lambda t: 1.0, lambda t: 3.0]
f_array = [lambda t: 0.0]

gamma_array = np.arange(0.001,0.1,0.02)
gamma_array = [0.1]

# delta_array = np.arange(0.0,1.0,0.3)
delta_array = [0.0]

t = np.arange(0,10.0,0.01)
btw = 1

# g_array = np.arange(0.0,1.0,0.3)
g_array = [0.01,0.0]
y0 = []

# for x in np.arange(-30.0,30.0,5):
#     for y in np.arange(-30.0,30.0,5):
#         y0.append((x,y))

y0 = [ (1.0,0.0)]
n = 1000

pro = []

MAXP = 6

for g in g_array:
    for f in f_array:
        for gamma in gamma_array:
            for delta in delta_array:
                if len(pro) == MAXP:
                    i = 0
                    while True:
                        if not pro[i%MAXP].is_alive():
                            del pro[i%MAXP]
                            break
                        i+=1
                print "======== %d ======="%len(pro)
                p = Process(target=run,args=(y0,t,btw,delta,g,gamma,f,n))
                p.start()
                pro.append(p)


for p in pro:
    p.join()
                         
    

