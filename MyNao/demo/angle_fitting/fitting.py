import os
import numpy as np 

root = 'C:\\Users\\lenovo\\Desktop\\AI-Project-Portfolio\\danceprimitives\\0000'

file = np.load(os.path.join(root, 'dance_motion_0.npy'))
print(file, file.shape)