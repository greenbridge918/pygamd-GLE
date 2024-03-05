import pygamd
from sys import argv
import numpy as np

mst = pygamd.snapshots.mst_parser.read_mst("output.mst")
vel = np.asarray(mst.velocity)
n = mst.nframes

vel = vel.reshape(n, 2000, 3)

vel = vel/np.sqrt(104)

def acf(allX1, allX2):
    M=allX1.shape[0]
    return(np.fft.irfft(np.sum(abs(np.multiply(np.fft.rfft(allX1, axis=0, n=M*2), np.conj(np.fft.rfft(allX2, axis=0, n=M*2)))), axis=-1), axis=0,n=2*M)[:M,:].real/(np.arange(M,0,-1
)[:,None]))

vacf = (acf(vel, vel)).T.mean(axis=0)
times = []
dt = float(argv[1]) # unit ps
frames = range(vel.shape[0])
for frame in frames:
    time = frame * dt
    times.append(time)
times = np.array(times)
np.savetxt("VACF.txt", np.c_[times,vacf], fmt = '%.6f')

