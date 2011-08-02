import numpy as np
import pylab as pl
import save
import os

filename = os.sys.argv[1]

meta, t, plots = save.load(filename)

var = np.zeros_like(plots[0])

for p in plots:
    var += p**2

var = var/len(plots)

pl.plot(t,var)
pl.show()


mean = np.zeros_like(plots[0])
for p in plots:
    mean += p

mean /= len(plots)

pl.plot(t,mean)
pl.show()
