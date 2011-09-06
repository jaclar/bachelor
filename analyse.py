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

# Gesamten Phasenraum plotten (a)
ar_avg = np.zeros_like(plots[0][:,0])
ai_avg = np.zeros_like(plots[0][:,0])
for p in plots:
    ar = p[:,0]
    ai = p[:,1]
    ar_avg += ar
    ai_avg += ai
    pl.plot(ar,ai)
    pl.title("Phasenraum (Gesamt)")
    pl.xlabel("$\Re a$")
    pl.ylabel("$\Im a$")
pl.show()

# Gesamten Phasenraum plotten (a^*)
acr_avg = np.zeros_like(plots[0][:,0])
aci_avg = np.zeros_like(plots[0][:,0])
for p in plots:
    acr = p[:,2]
    aci = p[:,3]
    acr_avg += acr
    aci_avg += aci
    pl.plot(acr,aci)
    pl.title("Phasenraum (Gesamt)")
    pl.xlabel("$\Re a^*$")
    pl.ylabel("$\Im a^*$")
pl.show()

ar_avg /= len(plots)
ai_avg /= len(plots)
acr_avg /= len(plots)
aci_avg /= len(plots)

# Durchschnittliche Trajektorien
pl.plot(ar_avg,ai_avg)
pl.plot(acr_avg,-1.0*aci_avg)
pl.show()

# Phasenraumabstand zwischen der komplex konjugierten Variablen
pl.plot(t,np.sqrt((ar_avg-acr_avg)**2+(ai_avg+aci_avg)**2))
pl.title("abstand")
pl.show()

# Photonenanzahl
n_avg = np.zeros_like(t)
for p in plots:
    n = (p[:,0]+1.0j*p[:,1])*(p[:,2]+1.0j*p[:,3])
    n_avg += n.real
    pl.plot(t,n.real)
    pl.ylabel("$a^*a=n$")
    pl.xlabel("$t$")
pl.show()

# Durchschnittliche Photonenanzahl
pl.ylabel("$a^*a=n$")
pl.xlabel("$t$")
pl.plot(t,n_avg/len(plots))
pl.show()
