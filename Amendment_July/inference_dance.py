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

class actuator(object):
    def __init__(self):
        self.ip = '127.0.0.1'
        self.port = 19997
        sim.simxFinish(-1)  # just in case, close all opened connections
        self.clientID = sim.simxStart(self.ip, self.port, True, True, -5000, 5)
        # Connect to V-REP
        if self.clientID == -1:
            import sys
            sys.exit('\nV-REP remote API server connection failed (' + self.ip + ':' + str(self.port) + '). Is V-REP running?')
        print('Connected to Remote API Server')  # show in the terminal
        sim.simxStartSimulation(self.clientID, sim.simx_opmode_oneshot)
        self.Body = {}
        get_first_handles(self.clientID, self.Body)    
        #errorCode, acc = sim.simxGetStringSignal(self.clientID, 'Acceleration', sim.simx_opmode_streaming)
        #returnCode, position = sim.simxGetObjectPosition(self.clientID, self.Body['HeadYaw'], -1, sim.simx_opmode_streaming)
    
    def actuate(self, primitive, converter):
        for idx_f in range(primitive.shape[0]):
            angle_recon = converter.frameRecon(primitive[idx_f])
            # angles: LSP, LSR, LEY, LER, RSP, RSR, REY, RER, LHP, LHR, LHYP, LKP, RHP, RHR, RHYP, RKP, LAP, LAR, RAP, RAR
            angles = converter.generateWholeJoints(angle_recon)
            assert(len(angles)==20)
            transmit(self.clientID, self.Body, angles)
            time.sleep(0.03)
            #returnCode, position=sim.simxGetObjectPosition(self.clientID, Body['HeadYaw'], -1, sim.simx_opmode_buffer)



if __name__ == '__main__':  
    modelR = Relevance(space_dims=17*32, hidden_dims=2048, relevance_dims=128)
    #model.load_state_dict(torch.load('./params/Thu Jul 16 16-38-44 2020@/best_fitted_params.pt')['model_state_dict'])
    params = torch.load('./params/Thu Jul 16 16-38-44 2020@/best_fitted_params.pt')['model_state_dict']
    new_params = OrderedDict()
    for k, v in params.items():
        name = k[7:]
        new_params[name] = v
    modelR.load_state_dict(new_params)
    modelR.eval()

    modelC = CapabilityWithAttention(time_dim=32)
    modelC.load_state_dict(torch.load('./params-capability/Sun Oct 11 13-13-49 2020/best_fitted_params.pt')['model_state_dict'])
    #modelC = Capability_Naive(space_dims=17*32, hidden_dims=1024, representation_dim=128)
    #modelC.load_state_dict(torch.load('./params-capability/Fri Jul 17 01-17-05 2020/best_fitted_params.pt')['model_state_dict'])
    modelC.eval()

    loader = inferenceDataLoader('./dance_unit_data.npy')
    #loader.get_encoded_library(modelR, 1024)
    loader.load_encoded_library()

    converter = coordinate2angle()
    converter.set_bound(np.load('bound_range.npy'), np.load('step.npy'), 32, 16)
    
    actuator = actuator()
    
    current = 33
    print('current primitive:', current)
    while True:
        idx, data = loader.pick_data(False, current)
        actuator.actuate(data[0], converter)
        data_encoded = modelR(data).detach().numpy()
        result = loader.search(data_encoded)[1:]
        if result[0][0] == idx-1:
            result = result[1:]
        loader.check_stability([i[0] for i in result], data, modelC)
        a = [result[i][0] for i in range(len(result))]
        p = [result[i][1] for i in range(len(result))]
        p = p / sum(p)
        current = np.random.choice(a=a, p=p)
        print('current primitive:', current)

    