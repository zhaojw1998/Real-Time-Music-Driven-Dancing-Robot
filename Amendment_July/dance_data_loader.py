import torch.utils.data as data
from torch.utils.data import DataLoader
import numpy as np
import os

class danceUnitDataLodaer(data.Dataset):
    def __init__(self, data_path):
        self.data = np.load(data_path)
        #self.data = self.data / np.power(np.sum(np.power(self.data, 2), axis=-1)[:, :, np.newaxis], 0.5)
        self.space_dim = self.data.shape[-1]
        self.data_volumn = self.data.shape[0]
        tmp = np.concatenate((self.data[1:, :, :], self.data[:1, :, :]), axis=0)
        self.data = np.concatenate((self.data, tmp), axis = -1)
        np.random.shuffle(self.data)

    def __getitem__(self, index):
        origin = self.data[index, :, :self.space_dim] 
        target = self.data[index, :, self.space_dim:]
        r1 = np.random.randint(2, self.data_volumn)
        random_1 = self.data[(index+r1)%self.data_volumn, :, :self.space_dim]
        r2 = np.random.randint(2, self.data_volumn)
        random_2 = self.data[(index+r2)%self.data_volumn, :, :self.space_dim]
        r3 = np.random.randint(2, self.data_volumn)
        random_3 = self.data[(index+r3)%self.data_volumn, :, :self.space_dim]
        r4 = np.random.randint(2, self.data_volumn)
        random_4 = self.data[(index+r4)%self.data_volumn, :, :self.space_dim]
        return origin, target, random_1, random_2, random_3, random_4

    def __len__(self):
        return self.data_volumn

if __name__ == '__main__':
    train_dataloader = DataLoader(danceUnitDataLodaer('Amendment_july/dance_unit_data.npy'), batch_size = 4, shuffle = True, num_workers = 4, drop_last = True)
    for step, (origin, target, random_1, random_2, random_3, random_4) in enumerate(train_dataloader):
        print(random_3.shape)
        break