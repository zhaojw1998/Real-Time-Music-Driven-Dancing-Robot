from multiprocessing import Process, Queue
from angel_transmit import joint_actuate, init_joint
from manage_joints import get_first_handles
import pickle
import time
import numpy as np
import json
import os
import sim
import sys
from tqdm import tqdm

with open('Amendments/scipts for graph construction/graph construction 0516/03.stable_typical_sequence_redundant_supress.txt', 'rb') as f:
    stable_typical_sequence_redundant_supress = pickle.load(f)
sequence_head = []
sequence_tail = []
for sequence in stable_typical_sequence_redundant_supress:
    sequence_head.append(sequence[0])
    sequence_tail.append(sequence[-1])
with open('Amendments/scipts for graph construction/graph construction 0516/04.incidence_matrix_and_interface.txt', 'rb') as f:
    incidence_matrix = pickle.load(f)
    interface = pickle.load(f)
    state_idx_dict = pickle.load(f)
    state_set = pickle.load(f)
with open('Amendments/scipts for graph construction/graph construction 0516/06.transition_graph.txt', 'rb') as f:
    transition_graph = pickle.load(f)


ip = '127.0.0.1'
port = 19997
sim.simxFinish(-1)  # just in case, close all opened connections
clientID = sim.simxStart(ip, port, True, True, -5000, 5)
# Connect to V-REP
if clientID == -1:
    import sys
    sys.exit('\nV-REP remote API server connection failed (' + ip + ':' + str(port) + '). Is V-REP running?')
print('Connected to Remote API Server')  # show in the terminal

sim.simxStartSimulation(clientID, sim.simx_opmode_oneshot)
Body = {}
get_first_handles(clientID,Body)
returnCode, position=sim.simxGetObjectPosition(clientID, Body['HeadYaw'], -1, sim.simx_opmode_streaming)
returnCode, position=sim.simxGetObjectPosition(clientID, Body['LAnklePitch'], -1, sim.simx_opmode_streaming)
current_state = 0
mode = 0
layer = 0
current_state=[27, 28, 29, 30]
for i in range(len(current_state)):
    primitive = np.load('C:\\Users\\lenovo\\Desktop\\AI-Project-Portfolio\\danceprimitives\\'+'{0}'.format(str(current_state[i]).zfill(4))+'\\dance_motion_'+str(current_state[i])+'.npy')
    print('primitive:', current_state[i])
    primitive_backup = primitive.reshape(-1, 17, 3)
    primitive = primitive_backup
    motion = {}
    for i in range(primitive.shape[0]):
        motion[str(i)] = {}
        for j in range(primitive.shape[1]):
            motion[str(i)][str(j)] = [primitive[i][j][0], primitive[i][j][2], primitive[i][j][1]]
    for frame in range(primitive.shape[0]):
        print('\t'+'frame:' + str(frame))
        joint_actuate(clientID, Body, motion, frame)
        returnCode, position=sim.simxGetObjectPosition(clientID, Body['HeadYaw'], -1, sim.simx_opmode_buffer)
        time.sleep(0.2)
"""
while True:
    print(mode, current_state)
    returnCode, position_=sim.simxGetObjectPosition(clientID, Body['LAnklePitch'], -1, sim.simx_opmode_buffer)
    #print('LAnklePitch position:', position_)
    if mode == 0:
        primitive = np.load('C:\\Users\\lenovo\\Desktop\\AI-Project-Portfolio\\danceprimitives\\'+'{0}'.format(str(current_state).zfill(4))+'\\dance_motion_'+str(current_state)+'.npy')
        primitive_backup = primitive.reshape(-1, 17, 3)
        #primitive_revised = np.empty((primitive_backup.shape[0], primitive_backup.shape[1], primitive_backup.shape[2]))
        #primitive_revised[:, :, 0] = -primitive_backup[:, :, 1]
        #primitive_revised[:, :, 1] = primitive_backup[:, :, 0]
        #primitive_revised[:, :, 2] = -primitive_backup[:, :, 2]
        #primitive = primitive_revised.reshape(-1, 51)
        primitive = primitive_backup
        motion = {}
        for i in range(primitive.shape[0]):
            motion[str(i)] = {}
            for j in range(primitive.shape[1]):
                motion[str(i)][str(j)] = [primitive[i][j][0], primitive[i][j][2], primitive[i][j][1]]
        for frame in range(primitive.shape[0]):
            joint_actuate(clientID, Body, motion, frame)
            returnCode, position=sim.simxGetObjectPosition(clientID, Body['HeadYaw'], -1, sim.simx_opmode_buffer)
            time.sleep(0.01)

        current_state = np.random.choice(transition_graph[current_state]['next'], p=np.array(transition_graph[current_state]['p']).ravel())
        #print('current:', current_state)
        if current_state in sequence_head:
            mode = 1
            p = np.array(interface['entrance'])==state_idx_dict[current_state]
            layer = np.random.choice(interface['layer'], p=(np.divide(p, np.sum(p)).ravel()))
            continue
    if mode == 1:
        primitive = np.load('C:\\Users\\lenovo\\Desktop\\AI-Project-Portfolio\\danceprimitives\\'+'{0}'.format(str(current_state).zfill(4))+'\\dance_motion_'+str(current_state)+'.npy')
        primitive_backup = primitive.reshape(-1, 17, 3)
        #primitive_revised = np.empty((primitive_backup.shape[0], primitive_backup.shape[1], primitive_backup.shape[2]))
        #primitive_revised[:, :, 0] = -primitive_backup[:, :, 1]
        #primitive_revised[:, :, 1] = primitive_backup[:, :, 0]
        #primitive_revised[:, :, 2] = -primitive_backup[:, :, 2]
        #primitive = primitive_revised.reshape(-1, 51)
        primitive = primitive_backup
        motion = {}
        for i in range(primitive.shape[0]):
            motion[str(i)] = {}
            for j in range(primitive.shape[1]):
                motion[str(i)][str(j)] = [primitive[i][j][0], primitive[i][j][2], primitive[i][j][1]]
        for frame in range(primitive.shape[0]):
            joint_actuate(clientID, Body, motion, frame)
            returnCode, position=sim.simxGetObjectPosition(clientID, Body['HeadYaw'], -1, sim.simx_opmode_buffer)
            time.sleep(0.1)
        try:
            next_state = state_set[int(incidence_matrix[layer][state_idx_dict[current_state]])]
            current_state = next_state
        except ValueError:
            try:
                next_state = state_set[int(incidence_matrix[layer][state_idx_dict[current_state]].split('_')[0])]
                current_state = next_state
                layer +=1
            except ValueError:
                current_state = np.random.choice(transition_graph[current_state]['next'], p=np.array(transition_graph[current_state]['p']).ravel())
                if not current_state in sequence_head:
                    mode = 0
                continue
"""