import numpy as np
import pylab as pl
import datetime

def save(xi_var,xi_mean,gamma,delta,g,data):
    meta = np.array([xi_var,xi_mean,gamma,delta,g])
    timestamp = datetime.datetime.isoformat(datetime.datetime.now())
    np.savez("data-%s.npz"%timestamp,*data,meta=meta)
    for l in data[1:]:
        pl.subplot(211)
        pl.ylabel("$\Re a$")
        pl.plot(data[0],l[:,0])
        pl.subplot(212)
        pl.ylabel("$\Im a$")
        pl.xlabel("t")
        pl.plot(data[0],l[:,1])
    #pl.legend()
    pl.subplot(211)
    pl.title("Symetrisiert: \n  $\\sigma_\\xi = %f,\\; \\bar{\\xi} = %f,\\; \\gamma = %f,\\; \\delta = %f,\\; g = %f$"%(xi_var,xi_mean,gamma,delta,g))
    pl.savefig("pic-%s.png"%timestamp,format='png')


