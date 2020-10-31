import torch
import os
import json
import numpy as np
from model import Relevance, Capability_Naive, CapabilityWithAttention
from tqdm import tqdm
from collections import OrderedDict
from coordinate2angle import coordinate2angle
from manage_joints import get_first_handles
from transmit import transmit
import sim
import time

class inferenceDataLoader(object):
    def __init__(self, data_path):
        self.data = np.load(data_path)
        #self.data = self.data / np.power(np.sum(np.power(self.data, 2), axis=-1)[:, :, np.newaxis], 0.5)
        self.dataVolumn = self.data.shape[0]
        self.spaceDim = self.data.shape[-1]
        self.data = torch.from_numpy(self.data).float()
    
    def get_encoded_library(self, modelR, batch_size):
        self.library_encoded = np.empty((0, 128))
        self.library_encoded = torch.from_numpy(self.library_encoded).float()
        for i in tqdm(range(0, self.dataVolumn-self.dataVolumn%batch_size, batch_size)):
            dataBatch = self.data[i:i+batch_size, :, :]
            #print(dataBatch.shape)
            data_encode = modelR(dataBatch)
            #print(data_encode.shape)
            self.library_encoded = torch.cat((self.library_encoded, data_encode), dim=0)

        #print(self.library_encoded.shape)
        if self.library_encoded.shape[0] < self.dataVolumn:
            dataBatch = self.data[self.dataVolumn-self.dataVolumn%batch_size: self.dataVolumn, :, :]
            data_encode = modelR(dataBatch)
            self.library_encoded = torch.cat((self.library_encoded, data_encode), dim=0)
        self.library_encoded = self.library_encoded.detach().numpy()
        np.save('library_encoded.npy', self.library_encoded)
        #print(self.library_encoded.detach().numpy().shape)
    def load_encoded_library(self):
        self.library_encoded = np.load('library_encoded.npy')
    
    def pick_data(self, random=True, idx=0):
        if random == True:
            index = np.random.randint(0, self.dataVolumn-1)
        else:
            index = idx
        return index, self.data[index: index+1, :, :]

    def search(self, data):
        #print(self.library_encoded.shape, data.shape)
        result = np.dot(self.library_encoded, data[0])/(np.linalg.norm(self.library_encoded, axis=1) * np.linalg.norm(data))
        print(result.shape)
        top_idx=result.argsort()[::-1][0:10]
        return [(i, result[i]) for i in list(top_idx)]

    def check_stability(self, candidates, current, modelC):
        current = current.expand(len(candidates), current.shape[1], current.shape[2])
        next_unit = self.data[candidates]
        #print(current.shape, next_unit.shape)
        ouput = modelC(torch.cat((current, next_unit), dim=1)).detach().numpy()
        print(np.exp(ouput[:, 0])/(np.exp(ouput[:, 0]) + np.exp(ouput[:, 1])))

class Motion_Selector(object):
    def __init__(self, current):
        self.modelR = Relevance(space_dims=17*32, hidden_dims=2048, relevance_dims=128)
        #model.load_state_dict(torch.load('./params/Thu Jul 16 16-38-44 2020@/best_fitted_params.pt')['model_state_dict'])
        params = torch.load('./params/Thu Jul 16 16-38-44 2020@/best_fitted_params.pt')['model_state_dict']
        new_params = OrderedDict()
        for k, v in params.items():
            name = k[7:]
            new_params[name] = v
        self.modelR.load_state_dict(new_params)
        self.modelR.eval()

        self.modelC = CapabilityWithAttention(time_dim=32)
        self.modelC.load_state_dict(torch.load('./params-capability/Sun Oct 11 13-13-49 2020/best_fitted_params.pt')['model_state_dict'])
        #modelC = Capability_Naive(space_dims=17*32, hidden_dims=1024, representation_dim=128)
        #modelC.load_state_dict(torch.load('./params-capability/Fri Jul 17 01-17-05 2020/best_fitted_params.pt')['model_state_dict'])
        self.modelC.eval()

        self.loader = inferenceDataLoader('./dance_unit_data.npy')
        #loader.get_encoded_library(modelR, 1024)
        self.loader.load_encoded_library()

        self.converter = coordinate2angle()
        self.converter.set_bound(np.load('bound_range.npy'), np.load('step.npy'), 32, 16)
        self.current = current


    def transfer(self):
        idx, data = self.loader.pick_data(False, self.current)
        data_encoded = self.modelR(data).detach().numpy()
        result = self.loader.search(data_encoded)[1:]
        if result[0][0] == idx-1:
            result = result[1:]
        self.loader.check_stability([i[0] for i in result], data, self.modelC)
        a = [result[i][0] for i in range(len(result))]
        p = [result[i][1] for i in range(len(result))]
        p = p / sum(p)
        self.current = np.random.choice(a=a, p=p)
        return self.current

    def transmit_motion_info(self, queue):
        if not queue.empty():
            _ = queue.get()
        queue.put(self.current)
        #print('mode:', self.mode, 'current:', self.current)

    def show(self):
        print('current:', self.current)

if __name__ == '__main__':
    motion_selector = Motion_Selector(33)
    for i in range(20):
        motion_selector.show()
        current = motion_selector.transfer()
