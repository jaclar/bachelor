import numpy as np
import pylab as pl
import save
import os
from scipy.interpolate import Rbf

filename = os.sys.argv[1]

meta, t, plots = save.load(filename)

def smooth(p,n):
    X = np.zeros(int(len(p)/n))
    xi = np.linspace(0.0,t[-1],int(len(p)/n))
    for i in range(int(len(p)/n)):
        X[i] = np.sum(p[i*n:(i+1)*n]/n)
    xi += (xi[0]+xi[1])/2.0
    return (xi,X)

fig_width_pt = 441.0  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inch
golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height = fig_width*golden_mean      # height in inches
fig_size =  [fig_width,fig_height]

params = {'backend': 'ps',
          'axes.labelsize': 10,
          'text.fontsize': 10,
          'legend.fontsize': 10,
          'xtick.labelsize': 8,
          'ytick.labelsize': 8,
          'text.usetex': True,
          'figure.figsize': fig_size}
pl.rcParams.update(params)


nmean = np.zeros_like(t)
rmean = np.zeros_like(plots[0])
i = 0
for p in plots:
    #if (p[-1,0]**2 +  p[-1,1]**2) > 700:
    i +=1
    nmean += p[:,0]**2+p[:,1]**2
    rmean += p

nmean /= i
rmean /= i

pl.plot(t,nmean)
pl.title("Mittelwert")
pl.xlabel("$t$")
pl.ylabel("$\langle n \\rangle$")
pl.show()

nvar = np.zeros_like(t)
rvar = np.zeros_like(plots[0])


for p in plots:
    nvar += (p[:,0]**2 + p[:,1]**2)**2 - nmean**2
    rvar += p**2 - rmean**2

nvar = nvar/len(plots)
rvar /= len(plots)

pl.plot(t,nvar,color="lightgrey")
pl.title("Varianz n")
pl.xlabel("$t$")
pl.ylabel("$\langle n^2 \\rangle$")
snvar = smooth(nvar,400)
pl.plot(snvar[0],snvar[1])
pl.show()

pl.xlim(0.0,300.0)
pl.plot(t,rvar,color='lightgrey')
pl.xlabel("$t$")
pl.ylabel("$\langle \Delta x \\rangle, \langle \Delta p \\rangle$")
sxvar = smooth(rvar[:,0],400)
spvar = smooth(rvar[:,1],400)
pl.plot(sxvar[0],sxvar[1],'--k')
pl.plot(spvar[0],spvar[1],'k')

pl.show()

