import scipy.special as spc
import numpy as np
import pylab as plt


R = 15
M = 20

p = np.arange(M)
pi = (p/(2*np.e*M))**p * (p<R)
pi = pi/pi.sum()/2
pi[-1] = 1/2

print(pi.sum())
print(pi)

plt.figure(1)

plt.plot(p,pi, 'x--')
plt.title(r'Prior $\pi$')
plt.xlabel(r'$|p|$')
plt.ylabel('probability')
plt.show()
