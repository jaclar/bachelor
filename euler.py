import scipy as sp
import scipy.integrate as spi

import numpy as np
import pylab as pl

# Einzelner Zeitschritt
def euler_step(f,h,x,t,noise,args=()):
    t = np.array(t)
    x = np.array(x)
    # normaler Integrationsschritt
    k1 = h*f(x,t,*args)
    # Rauschterm generieren
    st = np.sqrt(h)*np.array([noise(*args) for i in np.ones_like(x)])
    return x + k1 + st

# Euler Integrationsfunktion
def euler(f,x0,t,btw,noise,args=()):
    # Anzahl der Schritte (btw = zwischnschritte)
    steps = len(t)*btw
    # Schrittlänge
    h = (t[1]-t[0])/btw

    # Dimension der DiffGleichung
    if np.isscalar(x0):
        lx = 1
    else:
        lx = len(x0)

    x = np.zeros((len(t),lx))
    x[0] = x0
    np.seterr(all='raise')

    # Trajektorie berechnen
    try:
        for i in range(1,steps):
            x[i//btw] = euler_step(f,h,x[(i-1)//btw],t[(i-1)//btw],noise,args=args)
    # Fehlerfall Divergenz abfangen
    except FloatingPointError:
        print "diverged :("
    return x

