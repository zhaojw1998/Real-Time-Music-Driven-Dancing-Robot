import torch
import os
import json
import numpy as np
from model import Relevance
from tqdm import tqdm

class inferenceDataLodaer(object):
    def __init__(self, data_path):
        self.data = np.load(data_path)
        #self.data = self.data / np.power(np.sum(np.power(self.data, 2), axis=-1)[:, :, np.newaxis], 0.5)
        self.dataVolumn = self.data.shape[0]
        self.spaceDim = self.data.shape[-1]
        self.data = torch.from_numpy(self.data).float().cuda()
    
    def get_encoded_library(self, model, batch_size):
        self.library_encoded = np.empty((0, 128))
        self.library_encoded = torch.from_numpy(self.library_encoded).float().cuda()
        for i in tqdm(range(0, self.dataVolumn-self.dataVolumn%batch_size, batch_size)):
            dataBatch = self.data[i:i+batch_size, :, :]
            #print(dataBatch.shape)
            data_encode = model(dataBatch)
            #print(data_encode.shape)
            self.library_encoded = torch.cat((self.library_encoded, data_encode), dim=0)

        #print(self.library_encoded.shape)
        if self.library_encoded.shape[0] < self.dataVolumn:
            dataBatch = self.data[self.dataVolumn-self.dataVolumn%batch_size: self.dataVolumn, :, :]
            data_encode = model(dataBatch)
            self.library_encoded = torch.cat((self.library_encoded, data_encode), dim=0)
        #print(self.library_encoded.cpu().detach().numpy().shape)
    
    def pick_data(self, random=True, idx=0):
        if random == True:
            index = np.random.randint(0, self.dataVolumn-1)
        else:
            index = idx
        return index, self.data[index: index+1, :, :]

    def search(self, data):
        data_copied = data.expand(self.dataVolumn, 128)
        #print(data_copied.shape)
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-8)
        result = cos(self.library_encoded, data_copied)
        #print(result.shape)
        return np.argpartition(result.cpu().detach().numpy(),-10)[-10:]

if __name__ == '__main__':
    model = Relevance(space_dims=17*32, hidden_dims=2048, relevance_dims=128).cuda()
    model.load_state_dict(torch.load('Amendment_July/params/Wed Jul 15 01-02-14 2020/best_fitted_params.pt')['model_state_dict'])
    model.eval()
    loader = inferenceDataLodaer('Amendment_July/dance_unit_data.npy')
    loader.get_encoded_library(model, 64)
    idx, data = loader.pick_data(False, 873)
    data_encoded = model(data)
    #print(data_encoded.shape)
    result = loader.search(data_encoded)
    print(result)