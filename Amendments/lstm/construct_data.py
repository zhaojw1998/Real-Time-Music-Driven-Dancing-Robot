import os
import torch
import numpy as np 
import sys
import pickle

save_root = 'C:\\Users\\lenovo\\Desktop\\AI-Project-Portfolio\\Amendments\\lstm\\sequence set'
with open('C:\\Users\\lenovo\\Desktop\\AI-Project-Portfolio\\Amendments\\scipts for graph construction\\00.primitive_compact.txt', 'rb') as f:
    total_sequence = pickle.load(f)
num = 0
num_primitive = len(total_sequence)
for count_range in range(2, 21):
    for index in range(len(total_sequence) - count_range):
        x = np.array(total_sequence[index: index + count_range]).reshape((-1,1))
        y = np.array([total_sequence[index + count_range]]).reshape((-1,1))
        x = torch.from_numpy(x).long()
        x = torch.zeros(count_range, num_primitive).scatter_(1, x, 1)
        y = torch.from_numpy(y).long()
        y = torch.zeros(1, num_primitive).scatter_(1, y, 1)
        sample = torch.cat((x, y), 0)
        with open(os.path.join(save_root, str(num)+'.txt'), 'wb') as f:
            pickle.dump(sample, f)
        num += 1


