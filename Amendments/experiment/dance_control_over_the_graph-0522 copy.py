from multiprocessing import Process, Queue
import pickle
import time
import numpy as np
import json
import os
import sys
from tqdm import tqdm

with open('Amendments/scipts for graph construction/graph construction 0522/03.stable_typical_sequences_symmetrical.txt', 'rb') as f:
    stable_typical_sequence = pickle.load(f)
sequence_head = []
sequence_tail = []
for sequence in stable_typical_sequence:
    sequence_head.append(sequence[0])
    sequence_tail.append(sequence[-1])
with open('Amendments/scipts for graph construction/graph construction 0522/04.nx5_matrix_and_interface_asymmetrical.txt', 'rb') as f:
    nx5_matrix = pickle.load(f)
    interface = pickle.load(f)
with open('Amendments/scipts for graph construction/graph construction 0522/04.nx5_matrix_and_interface.txt', 'rb') as f:
    nx5_matrix_2 = pickle.load(f)
    interface = pickle.load(f)
    #state_idx_dict = pickle.load(f)
    #state_set = pickle.load(f)
with open('Amendments/scipts for graph construction/graph construction 0522/06.transition_graph.txt', 'rb') as f:
    limited_transition_graph = pickle.load(f)
    inter_transition_graph = pickle.load(f)

save_file = 'C:/Users/lenovo/Desktop/AI-Project-Portfolio/Amendments/experiment/record/graph-record5.txt'
current_state = 0
if current_state in sequence_head:
    mode = 1
    p = np.array(np.array(sequence_head)==current_state)
    row = np.random.choice(interface, p=(p/np.sum(p)).ravel())
else:
    mode = 0
    row = 0
column = 0
count = 0
record = []
while len(record) <348:
    #print(mode, current_state, row, column)
    #returnCode, position_=sim.simxGetObjectPosition(clientID, Body['LAnklePitch'], -1, sim.simx_opmode_buffer)
    #print('LAnklePitch position:', position_)
    if mode == 0:
        record.append(current_state)
        primitive = np.load('C:\\Users\\lenovo\\Desktop\\AI-Project-Portfolio\\danceprimitives\\'+'{0}'.format(str(abs(current_state)).zfill(4))+'\\dance_motion_'+str(abs(current_state))+'.npy')
        primitive_backup = primitive.reshape(-1, 17, 3)
        #primitive_revised = np.empty((primitive_backup.shape[0], primitive_backup.shape[1], primitive_backup.shape[2]))
        #primitive_revised[:, :, 0] = -primitive_backup[:, :, 1]
        #primitive_revised[:, :, 1] = primitive_backup[:, :, 0]
        #primitive_revised[:, :, 2] = -primitive_backup[:, :, 2]
        #primitive = primitive_revised.reshape(-1, 51)
        primitive = primitive_backup
        if count < 3:
            count += 1
            current_state = np.random.choice(limited_transition_graph[current_state]['next'], p=np.array(limited_transition_graph[current_state]['p']).ravel())
        else:
            count =0
            current_state = np.random.choice(inter_transition_graph[current_state]['next'], p=np.array(inter_transition_graph[current_state]['p']).ravel())
            if current_state in sequence_head:
                mode = 1
                p = np.array(np.array(sequence_head)==current_state)
                row = np.random.choice(interface, p=(p/np.sum(p)).ravel())
                continue
    if mode == 1:
        record.append(current_state)
        primitive = np.load('C:\\Users\\lenovo\\Desktop\\AI-Project-Portfolio\\danceprimitives\\'+'{0}'.format(str(abs(current_state)).zfill(4))+'\\dance_motion_'+str(abs(current_state))+'.npy')
        primitive_backup = primitive.reshape(-1, 17, 3)
        primitive = primitive_backup

        
        if nx5_matrix[row][column+1] == -1:
            """change"""
            current_state = nx5_matrix_2[row][column]
            """change"""
            column = 0
            #if np.random.rand() > 0.5:
            #    current_state = np.random.choice(sequence_head)
            current_state = np.random.choice(inter_transition_graph[current_state]['next'], p=np.array(inter_transition_graph[current_state]['p']).ravel())
            if not current_state in sequence_head:
                mode = 0
            continue
        column += 1
        if column == 4:
            row += 1
            column = 0
        current_state = nx5_matrix[row][column]
print(record)
with open(save_file, 'wb') as f:
    pickle.dump(record, f, protocol=2)