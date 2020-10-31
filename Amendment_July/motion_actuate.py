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
        
class Motion_Actuator(object):
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
        self.loader = inferenceDataLoader('./dance_unit_data.npy')
        self.loader.load_encoded_library()
        self.converter = coordinate2angle()
        self.converter.set_bound(np.load('bound_range.npy'), np.load('step.npy'), 32, 16)
        self.init_time = time.time()
        self.i = 0
        self.time_sleep = 0.01
        #errorCode, acc = sim.simxGetStringSignal(self.clientID, 'Acceleration', sim.simx_opmode_streaming)
        #returnCode, position = sim.simxGetObjectPosition(self.clientID, self.Body['HeadYaw'], -1, sim.simx_opmode_streaming)
    
    def actuate(self, queue_primitive_from_selector, queue_primitive_from_actuator, queue_frame, queue_time_sleep):
        if not queue_primitive_from_selector.empty():
            current = queue_primitive_from_selector.get()
            num_frame = 32
            current_time = time.time()-self.init_time
            if not queue_primitive_from_actuator.empty():
                _ = queue_primitive_from_actuator.get()
            queue_primitive_from_actuator.put((current_time, num_frame, self.i))
            self.i +=1
            _, data = self.loader.pick_data(False, current)
            primitive = data[0]
            for idx_f in range(primitive.shape[0]):
                if not queue_frame.empty():
                    _ = queue_frame.get()
                queue_frame.put(idx_f)
                angle_recon = self.converter.frameRecon(primitive[idx_f])
                # angles: LSP, LSR, LEY, LER, RSP, RSR, REY, RER, LHP, LHR, LHYP, LKP, RHP, RHR, RHYP, RKP, LAP, LAR, RAP, RAR
                angles = self.converter.generateWholeJoints(angle_recon)
                assert(len(angles)==20)
                transmit(self.clientID, self.Body, angles)
                if not queue_time_sleep.empty():
                    self.time_sleep = queue_time_sleep.get()
                #time_transmit = time.time()-time_1
                time.sleep(max(0.0001, self.time_sleep))
                #returnCode, position=sim.simxGetObjectPosition(self.clientID, Body['HeadYaw'], -1, sim.simx_opmode_buffer)

class Motion_Actuator_old(object):
    def __init__(self):
        
        ip = '127.0.0.1'
        port = 19997
        sim.simxFinish(-1)  # just in case, close all opened connections
        self.clientID = sim.simxStart(ip, port, True, True, -5000, 5)
        if self.clientID == -1:
            import sys
            sys.exit('\nV-REP remote API server connection failed (' + ip + ':' + str(port) + '). Is V-REP running?')
        print('Connected to Remote API Server')
        
        with open('primitive_base/dance_primitive_library_interpole.json') as f:
            self.dance_primitive_library = json.load(f)
        
        self.Body = {}
        get_first_handles(self.clientID, self.Body)    #get first handles of Nao in the virtual environment
        self.joint_actuator = Joint_Actuator(self.clientID, self.Body)
        
        self.init_time = time.time()
        self.time_sleep = 0.01
        self.i = 0
    
    def actuate(self, queue_primitive_from_selector, queue_primitive_from_actuator, queue_frame, queue_time_sleep):
        if not queue_primitive_from_selector.empty():
            primitive = queue_primitive_from_selector.get()
            #print(primitive)
            num_frame = len(self.dance_primitive_library[str(primitive)])
            current_time = time.time()-self.init_time
            if not queue_primitive_from_actuator.empty():
                _ = queue_primitive_from_actuator.get()
            queue_primitive_from_actuator.put((current_time, num_frame, self.i))
            self.i +=1
            #print('frmae', num_frame)
            #t1=time.time()
            for frame in range(num_frame):
                #print(time.time()-t1)
                #t1=time.time()
                #time_1=time.time()
                if not queue_frame.empty():
                    _ = queue_frame.get()
                queue_frame.put(frame)
                self.joint_actuator.joint_actuate(primitive, frame)
                #time.sleep(0.005)
                if not queue_time_sleep.empty():
                    self.time_sleep = queue_time_sleep.get()
                #time_transmit = time.time()-time_1
                time.sleep(max(0.0001, self.time_sleep))