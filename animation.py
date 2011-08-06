import numpy as np
import pylab as pl
import kutta as ku
import save
import os
import datetime

filename = os.sys.argv[1]

frames = 500

meta, t, plots = save.load(filename)

dir = "animation"

# find maximum amplitude
max = []
min = []
for plot in plots:
    max.append([np.max(plot[:,0]), np.max(plot[:,1])])
    min.append([np.min(plot[:,0]), np.min(plot[:,1])])
max = np.array(max)
min = np.array(min)
max = [np.max(max[:,0]),np.max(max[:,1])]
min = [np.min(min[:,0]),np.min(min[:,1])]
print min, max

# initiate figure
fig = pl.figure(figsize=(5,abs((max[1]-min[1])/(max[0]-min[0]))*5))
ax = fig.add_subplot(111)
ax.set_autoscale_on(False)
ax.set_xlim(min[0],max[0])
ax.set_ylim(min[1],max[1])

l = len(plots[0])
stepsize = int(len(plots[0])/frames)
print l, stepsize
for i in range(0,l,stepsize):
    print i
    ax.cla()
    ax.set_autoscale_on(False)
    ax.set_xlim(min[0],max[0])
    ax.set_ylim(min[1],max[1])

    for p in range(len(plots)):
        ax.plot(plots[p][i-300:i+1,0],plots[p][i-300:i+1,1],)
        ax.plot(plots[p][i,0],plots[p][i,1],'o')
    fig.savefig("%s/%05d.png"%(dir,i),format='png',dpi=100)
