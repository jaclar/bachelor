import numpy as np
import pylab as pl
import datetime

def save(xi_var,xi_mean,gamma,delta,g,f,data,dir):
    meta = np.array([xi_var,xi_mean,gamma,delta,g,f])
    timestamp = datetime.datetime.isoformat(datetime.datetime.now())
    np.savez("%s/data-%s.npz"%(dir,timestamp),*data,meta=meta)
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
    pl.title("Symetrisiert: \n  $\\sigma_\\xi = %f,\\; \\bar{\\xi} = %f,\\; \\gamma = %f,\\; \\delta = %f,\\; g = %f\\; f = %f$"%(xi_var,xi_mean,gamma,delta,g,f(1)))
    pl.savefig("%s/pic-%s.png"%(dir,timestamp),format='png')
    pl.clf()

def load(filename):
    data = np.load(filename)
    meta = {}
    t = None
    plots = []
    for key in data.keys():
        if key == 'meta':
            meta['xi_var'] = data[key][0]
            meta['xi_mean'] = data[key][1]
            meta['gamma'] = data[key][2]
            meta['delta'] = data[key][3]
            meta['g'] = data[key][4]
            # taking older file versions into account
            # which didn't save f()
            if len(data[key]) == 6:
                meta['f'] = data[key][5]
        elif key == 'arr_0':
            t = data[key]
        else:
            plots.append(data[key])
            
    return meta, t, plots
