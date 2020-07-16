import pickle
import numpy as np
from matplotlib import pyplot as plt
from scipy import fftpack
from scipy.linalg import dft

with open('Amendments/scipts for graph construction/graph construction 0522/00.primitive_compact.txt', 'rb') as f:
    primitive_compact = pickle.load(f)

#print(len(primitive_compact), len(record))
with open('graph-record8.txt', 'rb') as f:
    record = pickle.load(f)

with open('pro-record8.txt', 'rb') as f:
    record2 = pickle.load(f)

length = 99
y1=np.zeros((348-length+1))
L = len(record)
for i in range(0, L-length):
    #print(i)
    #print(len(primitive_compact), len(record[i: i+16]))
    a = np.correlate(primitive_compact, record[i: i+length], 'valid')
    #print(a.shape)
    y1 += np.array(a)
y1/=(348-length+1)
m=dft(len(y1))
b1=m@np.array(a)
x = np.arange(0, len(y1))
#plt.axis([0, 20, 0, 3])

y2=np.zeros((348-length+1))
L = len(record)
for i in range(0, L-length):
    #print(i)
    #print(len(primitive_compact), len(record[i: i+16]))
    a = np.correlate(primitive_compact, record2[i: i+length], 'valid')
    #print(a.shape)
    y2 += np.array(a)
y2/=(348-length+1)
m=dft(len(y2))
b2=m@np.array(y2)
#x = np.arange(0, len(y1))

plt.title('Result Analysis')
plt.plot(x, y1, color='green', label='ours')
plt.plot(x, y2,  color='skyblue', label='baseline')

#plt.plot(x, np.abs(y), color="r", linestyle="-", linewidth=1)
plt.show()
