from matplotlib import pyplot as plt
from scipy import interpolate
import numpy as np

a = np.loadtxt("AA_VACF.txt").T
b = np.loadtxt("VACF.txt").T


plt.plot(a[0], a[1], label="AA")
plt.plot(b[0], b[1], label="CG")

plt.xscale("log")
plt.legend()
plt.show()
