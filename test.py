import numpy as np 
import pickle
"""
a=np.array([[1,2,3], [4,5,6]])
b=np.array([[1,5,3], [4,7,6]])
c=np.array([[1,2,6], [2,5,6]])

d=[a,b,c]

with open('test.txt', 'wb') as f:
    pickle.dump(d, f)
"""
with open('test.txt', 'rb') as f:
    data = pickle.load(f)

print(data)
