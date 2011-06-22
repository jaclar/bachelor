import numpy as np
import pylab as pl
import kutta as ku
import save
import os

filename = os.sys.argv[1]

meta, t, plots = save.load(filename)

for p in plots:
    pl.plot(p[:,0],p[:,1])

pl.show()
