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
    """
    Langevin-Gleichung
    """
    a = x[0] + 1.0j*x[1]
    ac = x[2] + 1.0j*x[3]
    
    # DGL fuer a
    r = 1.0j*delta*a \
        - gamma*a \
        + f  \
        -1.0j*g*ac*a*a 
    # DGL fuer a*
    rc = -1.0j*delta*ac \
        - gamma*ac \
        + f.conjugate()  \
        + 1.0j*g*ac*ac*a 
    return np.array([r.real,r.imag,rc.real,rc.imag],dtype=np.float64)

def sqD(g,gamma,a):
    """
    Wurzel aus Matrix D generieren
    """
    return linalg.sqrtm([[-1.0j*g*a**2,2*gamma],
                         [2*gamma,1.0j*g*a.conjugate()**2]])

def run(y0, t, btw, delta, g, gamma, f, a0, n = 1):
    """
    Hauptroutine zur Generierung von n Trajektorien fuer alle
    Startwerte in Array y0
    """
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
    
    # Iteration ueber alle Startwerte
    for y in y0:
        # Anzahl der Realisierungen
        for i in range(n):
            xi1 = sp.random.normal(xi_mean,xi_var,int(len(t)*btw))
            xi2 = sp.random.normal(xi_mean,xi_var,int(len(t)*btw))
            print "random numbers generated",np.mean(xi1),np.mean(xi2)

            print y
            k = eu.euler(lan,y,t,btw,D,xi1,xi2, \
                            args=(delta,g,gamma,f))
            data.append(k)
    save.save(xi_var,xi_mean,gamma,delta,g,f,data,"data")

################################
# Initialisieren der Parameter #
################################

f_array = [50.0]

gamma_array = np.arange(0.001,0.1,0.02)
gamma_array = [0.012]

delta_array = [7.44]

t = np.arange(0,.1,0.001)
btw = 1

g_array = [0.009]
y0 = [ (2.0,-32,2.0,32.0)]
n = 5000


pro = []

MAXP = 6

a0 = 0.24051713-31.65591378j

# Iteration ueber verschiedene Parameter
for g in g_array:
    for f in f_array:
        for gamma in gamma_array:
            for delta in delta_array:
                # maximal MAXP Prozesse starten
                if len(pro) == MAXP:
                    i = 0
                    while True:
                        # Falls ein Prozess beendet ist, kann ein
                        # neuer gestartet werden
                        if not pro[i%MAXP].is_alive():
                            del pro[i%MAXP]
                            break
                        i+=1
                print "======== %d ======="%len(pro)
                # neuen Prozess starten
                p = Process(target=run,args=(y0,t,btw,delta,g,gamma,f,a0,n))
                p.start()
                pro.append(p)

# Warten auf Beendigung aller Prozesse
for p in pro:
    p.join()
                         
    

