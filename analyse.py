import numpy as np
import pylab as pl
import kutta as ku
import save
import os
import correlation as co

filename = os.sys.argv[1]

meta, t, plots = save.load(filename)

for p in plots:
    pl.plot(p[:,0],p[:,1])
pl.show()

for p in plots:
    pl.plot(p[:,0]**2+p[:,1]**2)

pl.show()

ml = 2000
for p in plots:
    # pl.acorr(p[:,0]+1j*p[:,1],maxlags=ml)
    # pl.show()
    # pl.acorr(p[:,0],maxlags=ml,color="blue")
    # pl.acorr(p[:,1],maxlags=ml,color="red")
    # pl.show()
    x, y = co.icor(p,maxlags=ml,plot="phase")
    pl.subplot(211)
    pl.title("$\langle a(t)a(t)\\rangle$")
    pl.plot(x,np.real(y))
    pl.subplot(212)
    pl.plot(x,np.imag(y))
    pl.show()
    x, iy = co.icor(p,cc=True,maxlags=ml,plot="phase")
    pl.subplot(211)
    pl.title("$\langle a(t)a^*(t)\\rangle$")
    pl.plot(x,np.real(iy))
    pl.subplot(212)
    pl.plot(x,np.imag(iy))
    pl.show()
    
