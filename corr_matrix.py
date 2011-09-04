import numpy as np
import pylab as pl
import kutta as ku
import save
import os

filename = os.sys.argv[1]

meta, t, plots = save.load(filename)


for key,value in meta.items():
    print key, ":", value

a2_avg = np.zeros(len(t),dtype=np.complex128)
a_avg2 = np.zeros(len(t),dtype=np.complex128)
ac2_avg = np.zeros(len(t),dtype=np.complex128)
ac_avg2 = np.zeros(len(t),dtype=np.complex128)
acam_avg = np.zeros(len(t),dtype=np.complex128)
aca_avg = np.zeros(len(t),dtype=np.complex128)

for p in plots:
    a = p[:,0]+1.0j*p[:,1]
    # ac = p[:,2]+1.0j*p[:,3]
    ac = p[:,0]-1.0j*p[:,1]
    a2_avg += a**2
    a_avg2 += a

    acam_avg += ac*a

    ac2_avg += ac**2
    ac_avg2 += ac

a2_avg /= len(plots)
a_avg2 /= len(plots)
ac2_avg /= len(plots)
ac_avg2 /= len(plots)
acam_avg /= len(plots)
aca_avg /= len(plots)
 
print "a2_avg", a2_avg
print "a_avg2", a_avg2
print "ac2_avg", ac2_avg
print "ac_avg2", ac_avg2
print "acam_avg", acam_avg

for i in range(0,100,10):
    print i
    
    C = np.array([[ a2_avg[i] - a_avg2[i]**2,   
                    acam_avg[i] - a_avg2[i]*ac_avg2[i]],
                  [ acam_avg[i] - a_avg2[i]*ac_avg2[i],
                    ac2_avg[i] - ac_avg2[i]**2 ]])

    print C
    print
