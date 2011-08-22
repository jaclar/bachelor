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

fig_width_pt = 441.0  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inch
golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
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


pl.xlim(-25,25)
pl.ylim(-35,15)
pl.axvline(x=0,color='black')
pl.axhline(y=0,color='black')
for p in plots:
    pl.plot(p[:,0],p[:,1],'k')
    # pl.title("Phasenraum (Gesamt)")
    pl.xlabel("$p$")
    pl.ylabel("$x")
pl.show()


n_avg = np.zeros_like(t)
for p in plots:
    n_avg += p[:,0]**2+p[:,1]**2
    pl.plot(t,p[:,0]**2+p[:,1]**2)
    pl.ylabel("$a^*a=n$")
    pl.xlabel("$t$")
    # pl.title("Moden")
# pl.plot(t,n_avg/len(plots))
pl.show()


print len(plots) 

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

nmeta, nt, nplots = save.load('data/data-2011-08-21T20:54:15.632081.npz')


pl.ylabel("$a^*a=n$")
pl.xlabel("$t$")
pl.ylim(0.0,1.4)
pl.plot(t,n_avg/len(plots),'k')
pl.plot(t,nplots[0],'--k')

pl.show()


ml = 2000
av_corr = np.zeros_like(t,dtype=np.complex)
i = 0
n = 0
for p in plots:
    # pl.acorr(p[:,0]+1j*p[:,1],maxlags=ml)
    # pl.show()
    # pl.acorr(p[:,0],maxlags=ml,color="blue")
    # pl.acorr(p[:,1],maxlags=ml,color="red")
    # pl.show()

    # print p[0,0],p[0,1]
    # pl.plot(p[:,0],p[:,1])
    # pl.xlabel("$\Re a$")
    # pl.ylabel("$\Im a$")
    # pl.title("Phasenraum")
    # pl.show()
    # x, y = co.icor(p,maxlags=ml,plot="phase")
    # pl.subplot(211)
    # pl.title("$\langle a(t)a(t)\\rangle$")
    # pl.plot(x,np.real(y))
    # pl.subplot(212)
    # pl.plot(x,np.imag(y))
    # pl.show()
    # x, iy = co.icor(p,cc=True,maxlags=ml,plot="phase")
    # pl.subplot(211)
    # pl.title("$\langle a(t)a^*(t)\\rangle$")
    # pl.plot(x,np.real(iy))
    # pl.subplot(212)
    # pl.plot(x,np.imag(iy))
    # pl.show()
    n += 1
    if (p[-1,0]**2 +  p[-1,1]**2) > 700:
        x = p[:,0] + 1.0j*p[:,1]
        # corr = np.correlate(x,x,"same")
        # pl.plot(np.real(corr))
        # pl.plot(np.imag(corr))
        # pl.show()

        av_corr += corr
        i += 1
        print i,n

av_corr/= len(plots)
pl.plot(np.real(av_corr))
pl.plot(np.imag(av_corr))
pl.show()

#periodisch machen
av_corr = av_corr*np.sin(np.linspace(0.0,np.pi,len(av_corr)))**2

fft = np.fft.fft(av_corr)
pl.plot(np.real(fft))
pl.plot(np.imag(fft))
pl.show()
