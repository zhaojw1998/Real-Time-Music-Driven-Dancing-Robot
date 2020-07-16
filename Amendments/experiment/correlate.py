import pickle
import numpy as np
from matplotlib import pyplot as plt
from scipy import fftpack
from scipy.linalg import dft

with open('Amendments/scipts for graph construction/graph construction 0522/00.primitive_compact.txt', 'rb') as f:
    primitive_compact = pickle.load(f)

#print(len(primitive_compact), len(record))

def cxcorr(a,v):
    nom = np.linalg.norm(a[:])*np.linalg.norm(v[:])
    return fftpack.irfft(fftpack.rfft(a)*fftpack.rfft(v[::-1]))/nom

z=np.zeros((348,), dtype='complex128')
y=np.zeros((348,), dtype='complex128')
#for i in range(3):
with open('graph-record9.txt', 'rb') as f:
    record = pickle.load(f)
a=cxcorr(primitive_compact, record)
m=dft(348)
b=m@np.array(a)
#b = np.fft.fft(a)
#print(b)
y += b
z += a
with open('graph-record10.txt', 'rb') as f:
    record = pickle.load(f)
a=cxcorr(primitive_compact, record)
m=dft(348)
b=m@np.array(a)
#b = np.fft.fft(a)
#print(b)
y += b
z += a
with open('graph-record11.txt', 'rb') as f:
    record = pickle.load(f)
a=cxcorr(primitive_compact, record)
m=dft(348)
b=m@np.array(a)
#b = np.fft.fft(a)
#print(b)
y += b
z += a
with open('graph-record12.txt', 'rb') as f:
    record = pickle.load(f)
a=cxcorr(primitive_compact, record)
m=dft(348)
b=m@np.array(a)
#b = np.fft.fft(a)
#print(b)
y += b
z += a
with open('graph-record13.txt', 'rb') as f:
    record = pickle.load(f)
a=cxcorr(primitive_compact, record)
m=dft(348)
b=m@np.array(a)
#b = np.fft.fft(a)
#print(b)
y += b
z += a

y /= 5
z /= 5
x = np.arange(0, len(y))
#plt.axis([0, 20, 0, 3])
plt.plot(x[2:30], np.abs(y)[2:30], color="r", linestyle="-", linewidth=1)
plt.show()
#plt.plot(x, z, color="r", linestyle="-", linewidth=1)
#plt.show()