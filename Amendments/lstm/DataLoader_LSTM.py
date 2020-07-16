from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
import torch
import os
import cv2
import pickle 

#def default_loader(path):
#    Fx=np.load(os.path.join(path, 'Fx_freq_resized.npy'))
#    Fy=np.load(os.path.join(path, 'Fy_freq_resized.npy'))

#    return torch.from_numpy(Fx), torch.from_numpy(Fy)

class DataLoaderLSTM(Dataset):
    def __init__(self): #X, Y are npy file which contain file directories of train\val samples and corresponding labels
        self.save_root = 'C:\\Users\\lenovo\\Desktop\\AI-Project-Portfolio\\Amendments\\lstm\\sequence set'
    def __getitem__(self, index):
        with open(os.path.join(self.save_root, str(index)+'.txt'), 'rb') as f:
            file = pickle.load(f)

        return file

    def __len__(self):
        return len(os.listdir(self.save_root))