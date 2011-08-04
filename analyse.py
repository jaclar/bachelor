import numpy as np
import pylab as pl
import kutta as ku
import save
import os
import correlation as co

filename = os.sys.argv[1]

meta, t, plots = save.load(filename)

for key,value in meta.items():
    print key, ":", value

for p in plots:
    pl.plot(p[:,0],p[:,1])
    pl.title("Phasenraum (Gesamt)")
    pl.xlabel("$\Re a$")
    pl.ylabel("$\Im a$")
pl.show()


for p in plots:
    pl.plot(t,p[:,0]**2+p[:,1]**2)
    pl.ylabel("$|a|^2$")
    pl.xlabel("$t$")

pl.show()


ml = 2000
for p in plots:
    # pl.acorr(p[:,0]+1j*p[:,1],maxlags=ml)
    # pl.show()
    # pl.acorr(p[:,0],maxlags=ml,color="blue")
    # pl.acorr(p[:,1],maxlags=ml,color="red")
    # pl.show()
    print p[0,0],p[0,1]
    pl.plot(p[:,0],p[:,1])
    pl.xlabel("$\Re a$")
    pl.ylabel("$\Im a$")
    pl.title("Phasenraum")
    pl.show()
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
    
