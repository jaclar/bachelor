import numpy as np
import pylab as pl
import kutta as ku
import save
import os
import correlation as co

# Datei mit Trajektorien laden
filename = os.sys.argv[1]
meta, t, plots = save.load(filename)

# Metadaten ausgeben
for key,value in meta.items():
    print key, ":", value

# Latex Fromatierung
fig_width_pt = 441.0
inches_per_pt = 1.0/72.27
golden_mean = (np.sqrt(5)-1.0)/2.0
fig_height = fig_width
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

# Gesamten Phasenruam plotten
pl.xlim(-25,25)
pl.ylim(-35,15)
pl.axvline(x=0,color='black')
pl.axhline(y=0,color='black')
for p in plots:
    pl.plot(p[:,0],p[:,1],'k')
    pl.xlabel("$x$")
    pl.ylabel("$p$")
pl.show()

# Photonenzahl der einzelnen Trajektorien
n_avg = np.zeros_like(t)
for p in plots:
    n_avg += p[:,0]**2+p[:,1]**2
    pl.plot(t,p[:,0]**2+p[:,1]**2)
    pl.ylabel("$a^*a=n$")
    pl.xlabel("$t$")
pl.show()

# Latex Formatierung
print len(plots) 
fig_width_pt = 441.0
inches_per_pt = 1.0/72.27
golden_mean = (np.sqrt(5)-1.0)/2.0
fig_width = fig_width_pt*inches_per_pt
fig_height = fig_width*golden_mean
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

# Referenzdaten laden
nmeta, nt, nplots = save.load('data/data-2011-08-21T20:54:15.632081.npz')

# Durchschnittliche Photonenzahl laden
pl.ylabel("$a^*a=n$")
pl.xlabel("$t$")
pl.ylim(0.0,1.4)
pl.plot(t,n_avg/len(plots),'k',label="mit Rauschen")
pl.plot(t,nplots[0][:,0]**2+nplots[0][:,1]**2,'--k',label="ohne Rauschen")
pl.legend(loc=3)
pl.show()
