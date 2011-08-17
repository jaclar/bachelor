import numpy as np
import pylab as pl
import save
import os

filename = os.sys.argv[1]

meta, t, plots = save.load(filename)

# mean = np.zeros_like(plots[0])
# i = 0
# for p in plots:
#     #if (p[-1,0]**2 +  p[-1,1]**2) > 700:
#     i +=1
#     mean += p

# mean /= i

# pl.plot(t,mean)
# pl.title("Mittelwert")
# pl.xlabel("$t$")
# pl.ylabel("$\langle x \\rangle$")
# pl.show()

var = np.zeros_like(t)

for p in plots:
    var += p[:,0]**2 + p[:,1]**2 - 51.34387

var = var/len(plots)

pl.plot(t,var)
pl.title("Varianz")
pl.xlabel("$t$")
pl.ylabel("$\langle x^2 \\rangle$")
pl.show()

