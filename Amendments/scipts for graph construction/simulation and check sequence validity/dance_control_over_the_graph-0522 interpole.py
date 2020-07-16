from multiprocessing import Process, Queue
from angel_transmit import joint_actuate, init_joint, actuate
from angel_set_copy import set_angel
from manage_joints import get_first_handles
import pickle
import time
import numpy as np
import json
import os
import sim
import sys
from tqdm import tqdm
from scipy import interpolate

with open('Amendments/scipts for graph construction/graph construction 0522/03.stable_typical_sequences_symmetrical.txt', 'rb') as f:
    stable_typical_sequence = pickle.load(f)
sequence_head = []
sequence_tail = []
for sequence in stable_typical_sequence:
    sequence_head.append(sequence[0])
    sequence_tail.append(sequence[-1])
with open('Amendments/scipts for graph construction/graph construction 0522/04.nx5_matrix_and_interface.txt', 'rb') as f:
    nx5_matrix = pickle.load(f)
    interface = pickle.load(f)
    #state_idx_dict = pickle.load(f)
    #state_set = pickle.load(f)zz
with open('Amendments/scipts for graph construction/graph construction 0522/06.transition_graph.txt', 'rb') as f:
    limited_transition_graph = pickle.load(f)
    inter_transition_graph = pickle.load(f)


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
#returnCode, position=sim.simxGetObjectPosition(clientID, Body['HeadYaw'], -1, sim.simx_opmode_streaming)
#returnCode, position=sim.simxGetObjectPosition(clientID, Body['LAnklePitch'], -1, sim.simx_opmode_streaming)
current_state = 0
mode = 0
row = 0
column = 0
count = 0
while True:
    print(mode, current_state, row, column)
    #returnCode, position_=sim.simxGetObjectPosition(clientID, Body['LAnklePitch'], -1, sim.simx_opmode_buffer)
    #print('LAnklePitch position:', position_)
    if mode == 0:
        primitive = np.load('C:\\Users\\lenovo\\Desktop\\AI-Project-Portfolio\\danceprimitives\\'+'{0}'.format(str(abs(current_state)).zfill(4))+'\\dance_motion_'+str(abs(current_state))+'.npy')
        total = len(os.listdir('C:\\Users\\lenovo\\Desktop\\AI-Project-Portfolio\\danceprimitives'))
        next_primitive = np.load('C:\\Users\\lenovo\\Desktop\\AI-Project-Portfolio\\danceprimitives\\'+'{0}'.format(str((abs(current_state)+1)%total).zfill(4))+'\\dance_motion_'+str((abs(current_state)+1)%total)+'.npy').reshape(-1, 17, 3)[0, :, :]
        primitive_backup = primitive.reshape(-1, 17, 3)
        #primitive_revised = np.empty((primitive_backup.shape[0], primitive_backup.shape[1], primitive_backup.shape[2]))
        #primitive_revised[:, :, 0] = -primitive_backup[:, :, 1]
        #primitive_revised[:, :, 1] = primitive_backup[:, :, 0]
        #primitive_revised[:, :, 2] = -primitive_backup[:, :, 2]
        #primitive = primitive_revised.reshape(-1, 51)
        primitive = np.concatenate((primitive_backup, next_primitive[np.newaxis, :, :]), axis=0)
        motion = {}
        for i in range(primitive.shape[0]):
            motion[str(i)] = {}
            for j in range(primitive.shape[1]):
                motion[str(i)][str(j)] = [-primitive[i][j][2], primitive[i][j][0], -primitive[i][j][1]]
        angles = np.empty((len(motion), 20))
        for frame in range(len(motion)):
            angle = np.array(set_angel(motion, frame, current_state))
            angles[frame, :] = angle
        time_axis=np.arange(0, len(motion))
        #print(time_axis.shape, angles.shape)
        f = interpolate.interp1d(time_axis, angles, axis=0, kind='cubic')
        time_new = np.arange(0, len(motion)-1, len(motion)/50)
        angles_new = f(time_new)
        
        #a=0
        for i in range(angles_new.shape[0]):
            actuate(clientID, Body, angles_new[i])
            #returnCode, position=sim.simxGetObjectPosition(clientID, Body['HeadYaw'], -1, sim.simx_opmode_buffer)
            time.sleep(0.01)
            #print(a)
            #a+=1
            """if position[2] < 0.4 and position[2] > 0:   #fall down
                #print('fall down')
                sim.simxStopSimulation(clientID, sim.simx_opmode_oneshot)
                time.sleep(.1)
                sys.exit()"""
        if count < 3:
            count += 1
            current_state = np.random.choice(limited_transition_graph[current_state]['next'], p=np.array(limited_transition_graph[current_state]['p']).ravel())
        else:
            count =0
            current_state = np.random.choice(inter_transition_graph[current_state]['next'], p=np.array(inter_transition_graph[current_state]['p']).ravel())
            if current_state in sequence_head:
                mode = 1
                p = np.array(sequence_head==current_state)
                row = np.random.choice(interface, p=(p/np.sum(p)).ravel())
                continue
    if mode == 1:
        primitive = np.load('C:\\Users\\lenovo\\Desktop\\AI-Project-Portfolio\\danceprimitives\\'+'{0}'.format(str(abs(current_state)).zfill(4))+'\\dance_motion_'+str(abs(current_state))+'.npy')
        total = len(os.listdir('C:\\Users\\lenovo\\Desktop\\AI-Project-Portfolio\\danceprimitives'))
        next_primitive = np.load('C:\\Users\\lenovo\\Desktop\\AI-Project-Portfolio\\danceprimitives\\'+'{0}'.format(str((abs(current_state)+1)%total).zfill(4))+'\\dance_motion_'+str((abs(current_state)+1)%total)+'.npy').reshape(-1, 17, 3)[0, :, :]
        primitive_backup = primitive.reshape(-1, 17, 3)
        #primitive_revised = np.empty((primitive_backup.shape[0], primitive_backup.shape[1], primitive_backup.shape[2]))
        #primitive_revised[:, :, 0] = -primitive_backup[:, :, 1]
        #primitive_revised[:, :, 1] = primitive_backup[:, :, 0]
        #primitive_revised[:, :, 2] = -primitive_backup[:, :, 2]
        #primitive = primitive_revised.reshape(-1, 51)
        primitive = np.concatenate((primitive_backup, next_primitive[np.newaxis, :, :]), axis=0)
        motion = {}
        for i in range(primitive.shape[0]):
            motion[str(i)] = {}
            for j in range(primitive.shape[1]):
                motion[str(i)][str(j)] = [-primitive[i][j][2], primitive[i][j][0], -primitive[i][j][1]]
        angles = np.empty((len(motion), 20))
        for frame in range(len(motion)):
            angle = np.array(set_angel(motion, frame, current_state))
            angles[frame, :] = angle
        time_axis=np.arange(0, len(motion))
        #print(time_axis.shape, angles.shape)
        f = interpolate.interp1d(time_axis, angles, axis=0, kind='cubic')
        time_new = np.arange(0, len(motion)-1, len(motion)/50)
        angles_new = f(time_new)
        for i in range(angles_new.shape[0]):
            actuate(clientID, Body, angles_new[i])
            #returnCode, position=sim.simxGetObjectPosition(clientID, Body['HeadYaw'], -1, sim.simx_opmode_buffer)
            time.sleep(0.01)
            """if position[2] < 0.4 and position[2] > 0:   #fall down
                #print('fall down')
                sim.simxStopSimulation(clientID, sim.simx_opmode_oneshot)
                time.sleep(.1)
                sys.exit()"""
        
        if nx5_matrix[row][column+1] == -1:
            column = 0
            current_state = np.random.choice(inter_transition_graph[current_state]['next'], p=np.array(inter_transition_graph[current_state]['p']).ravel())
            if not current_state in sequence_head:
                mode = 0
            continue
        column += 1
        if column == 4:
            row += 1
            column = 0
        current_state = nx5_matrix[row][column]
