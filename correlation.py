import numpy as np
import pylab as pl
import save

def icor(f,cc=False,maxlags=10,plot=None):
    ret = []
    norm = 1.0
    for t in range(-maxlags,maxlags):
        if t > 0:
            a = f[:-t]
            b = f[t:]
        if t < 0:
            a = f[-t:]
            b = f[:t]
        if t == 0:
            a = f
            b = f
        
        a = a[:,0] + 1j*a[:,1]
        if cc:
            b = b[:,0] - 1j*b[:,1]
        else:
            b = b[:,0] + 1j*b[:,1]
        
        s = np.sum(a*b)
        ret.append(s)
        if t == 0:
            norm = s

    ret = ret/norm
    if plot == 'real':
        pl.plot(range(-maxlags,maxlags),ret)
        pl.show()
    if plot == 'phase':
        pl.plot(np.real(ret),np.imag(ret))
        pl.show()
    return range(-maxlags,maxlags), ret
