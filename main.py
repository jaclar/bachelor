import numpy as np
import pylab as pl
import kutta as ku
import euler as eu
import scipy.integrate as integrate
import scipy.linalg as linalg
import scipy as sp
import save
from multiprocessing import Process

def lan(x,t,delta,g,gamma,f):
    a = x[0] + 1.0j*x[1]
    ac = x[2] + 1.0j*x[3]
    r = 1.0j*delta*a \
        - gamma*a \
        + f  \
        -1.0j*g*ac*a*a 
    rc = -1.0j*delta*ac \
        - gamma*ac \
        + f.conjugate()  \
        + 1.0j*g*ac*ac*a 
    return np.array([r.real,r.imag,rc.real,rc.imag],dtype=np.float64)

def dlan(x,t,delta,g,gamma,f,a0):
    a = x[0] + 1.0j*x[1]
    ac = x[2] + 1.0j*x[3]
    n = a0*a0.conjugate()
    r = (1.0j*delta-gamma-2.0j*g*n)*a \
        -1.0j*g*a0**2*ac
    rc = (-1.0j*delta-gamma+2.0j*g*n)*ac \
        + 1.0j*g*a0.conjugate()**2*a
    return np.array([r.real,r.imag,rc.real,rc.imag],dtype=np.float64)

def noise(delta,g,gamma,xi,f):
    # return 0
    return np.sqrt(gamma)*xi()

def sqD(g,gamma,a):
    return linalg.sqrtm([[-1.0j*g*a**2,2*gamma],
                         [2*gamma,-1.0j*g*a.conjugate()**2]])

def run(y0, t, btw, delta, g, gamma, f, a0, n = 1):
    print y0, delta, g, gamma
    data = [t]
    xi_mean = 0.0
    xi_var = 1.0

    D = sqD(g,gamma,a0)
    
    print D[0,:]
    print D[1,:]

    y0 = np.array(y0)
    if y0.shape == (2,):
        y0 = np.array([y0])

    for y in y0:
        for i in range(n):
            xi1 = sp.random.normal(xi_mean,xi_var,int(len(t)*btw))
            xi2 = sp.random.normal(xi_mean,xi_var,int(len(t)*btw))
            print "random numbers generated"

            print y
            # k = ku.RK4(lan,y,t,btw,noise, \
            #              args=(delta,g,gamma,xi,f))
            # k = integrate.odeint(lan,y,t, \
            #                        args=(delta,g,gamma,xi,f))
            k = eu.euler(lan,y,t,btw,D,xi1,xi2, \
                            args=(delta,g,gamma,f))

            # k = eu.euler(oup,y,t,btw,oup_noise, \
            #                  args=(2.0,0.0,1.0))
            data.append(k)
    save.save(xi_var,xi_mean,gamma,delta,g,f,data,"data")
        
# f_array = [lambda t: p for p in np.arange(0.0,1.0,0.3)]
f_array = [lambda t: 0.3,lambda t: 1.0, lambda t: 3.0]
f_array = [lambda t: 50.0]
f_array = [50.0]

gamma_array = np.arange(0.001,0.1,0.02)
gamma_array = [0.012]

# delta_array = np.arange(0.0,1.0,0.3)
delta_array = [7.44]

t = np.arange(0,.1,0.001)
btw = 100

# g_array = np.arange(0.0,1.0,0.3)
g_array = [0.009]
y0 = []

# for x in np.arange(-30.0,30.0,5):
#     for y in np.arange(-30.0,30.0,5):
#         y0.append((x,y))

y0 = [ (0.24051713,-31.65591378,0.24051713,31.65591378)]
# y0 = [ (0.9,-32,0.9,32.0)]
n = 10000


pro = []

MAXP = 6

a0 = 0.24051713-31.65591378j

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
                p = Process(target=run,args=(y0,t,btw,delta,g,gamma,f,a0,n))
                p.start()
                pro.append(p)


for p in pro:
    p.join()
                         
    

