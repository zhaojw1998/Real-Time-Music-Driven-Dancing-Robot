import pickle
import os
from matplotlib import pyplot as plt 

root = 'C:/Users/lenovo/Desktop/AI-Project-Portfolio/Amendments/experiment/buzhen'

with open(os.path.join(root, 'frame_no_handle.txt'), 'rb') as f:
    no_handle = pickle.load(f)
no_handle = [i*3 for i in no_handle]
with open(os.path.join(root, 'frame_no_interpole.txt'), 'rb') as f:
    frame_no_interpole = pickle.load(f)
frame_no_interpole = [i*3 for i in frame_no_interpole]
with open(os.path.join(root, 'frame_interpole.txt'), 'rb') as f:
    frame_interpole = pickle.load(f)

x=list(range(200))

plt.title('Result Analysis')

plt.plot(x, no_handle, color='blueviolet', label='no alignment control')
plt.plot(x, frame_no_interpole,  color='magenta', label='alignment without interpolation')
plt.plot(x, frame_interpole, color='lightgreen', label='alignment wuth interpolation')
plt.legend() # 显示图例
plt.xlabel('beat')
plt.ylabel('alignment error')
plt.show()
