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

with open('Amendments/scipts for graph construction/graph construction 0522/05.adjacency_matrix_and_transition_states.txt', 'rb') as f:
    adjacency_matrix = pickle.load(f)
    transition_states = pickle.load(f)
with open('Amendments/scipts for graph construction/graph construction 0522/03.stable_typical_sequences_symmetrical.txt', 'rb') as f:
    stable_typical_sequences = pickle.load(f)
sequence_head = []
sequence_tail = []
for sequence in stable_typical_sequences:
    sequence_head.append(sequence[0])
    sequence_tail.append(sequence[-1])

print(np.sum(adjacency_matrix))
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

for idx_from in tqdm(range(adjacency_matrix.shape[0])):
    primitive_id = transition_states[idx_from]
    primitive = np.load('C:\\Users\\lenovo\\Desktop\\AI-Project-Portfolio\\danceprimitives\\'+'{0}'.format(str(abs(primitive_id)).zfill(4))+'\\dance_motion_'+str(abs(primitive_id))+'.npy')
    primitive_1 = primitive.reshape(-1, 17, 3)
    if primitive_id < 0:
        primitive_1[:, :, 0] = -primitive_1[:, :, 0]
    for idx_to in range(adjacency_matrix.shape[1]):
        if adjacency_matrix[idx_from][idx_to] == 1 or adjacency_matrix[idx_from][idx_to] == 2:
            primitive_id = transition_states[idx_to]
            primitive = np.load('C:\\Users\\lenovo\\Desktop\\AI-Project-Portfolio\\danceprimitives\\'+'{0}'.format(str(abs(primitive_id)).zfill(4))+'\\dance_motion_'+str(abs(primitive_id))+'.npy')
            primitive_2 = primitive.reshape(-1, 17, 3)
            if primitive_id < 0:
                primitive_2[:, :, 0] = -primitive_2[:, :, 0]
            primitive = np.concatenate((primitive_1, primitive_2), axis=0)
            
            motion = {}
            continue_flag = 0
            for i in range(primitive.shape[0]):
                motion[str(i)] = {}
                for j in range(primitive.shape[1]):
                    motion[str(i)][str(j)] = [-primitive[i][j][2], primitive[i][j][0], -primitive[i][j][1]]
            for frame in range(primitive.shape[0]):
                joint_actuate(clientID, Body, motion, frame, 1)
                returnCode, position=sim.simxGetObjectPosition(clientID, Body['HeadYaw'], -1, sim.simx_opmode_buffer)
                time.sleep(0.03)
                if position[2] < 0.4 and position[2] > 0:   #fall down
                    print(position[2], 'fall down at (', idx_from, ',', idx_to, ').')
                    adjacency_matrix[idx_from][idx_to] = 0
                    sim.simxStopSimulation(clientID, sim.simx_opmode_oneshot)
                    time.sleep(.1)
                    continue_flag = 1
                    break
            if continue_flag:
                sim.simxStartSimulation(clientID, sim.simx_opmode_oneshot)
                time.sleep(.1)
                continue
            sim.simxStopSimulation(clientID, sim.simx_opmode_oneshot)
            time.sleep(.1)
            sim.simxStartSimulation(clientID, sim.simx_opmode_oneshot)
            time.sleep(.1)
print(np.sum(adjacency_matrix))
to_delete = []
for i in range(adjacency_matrix.shape[0]):
    if transition_states[i] in sequence_head or transition_states[i] in sequence_tail:
        continue
    if all(adjacency_matrix[i] == 0):
        if not i in to_delete:
            to_delete.append(i)
for j in range(adjacency_matrix.shape[1]):
    if transition_states[j] in sequence_head or transition_states[j] in sequence_tail:
        continue
    if all(adjacency_matrix[:, j] == 0):
        if not j in to_delete:
            to_delete.append(j)
print(to_delete)
new_adjacency_matrix = np.empty((0, adjacency_matrix.shape[1]-len(to_delete)))
for i in range(adjacency_matrix.shape[0]):
    if i in to_delete:
        continue
    row_i = []
    for j in range(adjacency_matrix.shape[1]):
        if j not in to_delete:
            row_i.append(adjacency_matrix[i][j])
    new_adjacency_matrix = np.vstack((new_adjacency_matrix, np.array(row_i)))
assert(new_adjacency_matrix.shape[0] == new_adjacency_matrix.shape[1])

state_to_delete = [transition_states[i] for i in to_delete]
for state in state_to_delete:
    transition_states.remove(state)
assert(len(transition_states)==new_adjacency_matrix.shape[0])
print('states deleted:', state_to_delete)

with open('Amendments/scipts for graph construction/graph construction 0522/05.adjacency_matrix_and_transition_states_tested.txt', 'wb') as f:
    pickle.dump(new_adjacency_matrix, f, protocol=2)
    pickle.dump(transition_states, f, protocol=2)

with open('Amendments/scipts for graph construction/graph construction 0522/05.adjacency_matrix_and_transition_states_tested.txt', 'rb') as f:
    new_adjacency_matrix = pickle.load(f)
    transition_states = pickle.load(f)

print('check connectivity')
adjacency_matrix = new_adjacency_matrix
for idx_from in tqdm(range(adjacency_matrix.shape[0])):
    if not (transition_states[idx_from] in sequence_head or transition_states[idx_from] in sequence_tail):
        if sum(adjacency_matrix[idx_from]==1) < 1:
            idx_to_set = []
            while(len(idx_to_set) < 3):
                idx_to = np.random.randint(0, adjacency_matrix.shape[1])
                if transition_states[idx_to] in sequence_head or transition_states[idx_to] in sequence_tail or idx_to in idx_to_set:
                    continue
                primitive_id = transition_states[idx_from]
                primitive = np.load('C:\\Users\\lenovo\\Desktop\\AI-Project-Portfolio\\danceprimitives\\'+'{0}'.format(str(abs(primitive_id)).zfill(4))+'\\dance_motion_'+str(abs(primitive_id))+'.npy')
                primitive_1 = primitive.reshape(-1, 17, 3)
                if primitive_id < 0:
                    primitive_1[:, :, 0] = -primitive_1[:, :, 0]
                primitive_id = transition_states[idx_to]
                primitive = np.load('C:\\Users\\lenovo\\Desktop\\AI-Project-Portfolio\\danceprimitives\\'+'{0}'.format(str(abs(primitive_id)).zfill(4))+'\\dance_motion_'+str(abs(primitive_id))+'.npy')
                primitive_2 = primitive.reshape(-1, 17, 3)
                if primitive_id < 0:
                    primitive_2[:, :, 0] = -primitive_2[:, :, 0]
                primitive = np.concatenate((primitive_1, primitive_2), axis=0)
                motion = {}
                continue_flag = 0
                for i in range(primitive.shape[0]):
                    motion[str(i)] = {}
                    for j in range(primitive.shape[1]):
                        motion[str(i)][str(j)] = [-primitive[i][j][2], primitive[i][j][0], -primitive[i][j][1]]
                for frame in range(primitive.shape[0]):
                    joint_actuate(clientID, Body, motion, frame, 1)
                    returnCode, position=sim.simxGetObjectPosition(clientID, Body['HeadYaw'], -1, sim.simx_opmode_buffer)
                    time.sleep(0.03)
                    if position[2] < 0.4 and position[2] > 0:   #fall down
                        print(position[2], 'fall down at (', idx_from, ',', idx_to, ').')
                        #adjacency_matrix[idx_from][idx_to] = 0
                        sim.simxStopSimulation(clientID, sim.simx_opmode_oneshot)
                        time.sleep(.1)
                        continue_flag = 1
                        break
                if continue_flag:
                    sim.simxStartSimulation(clientID, sim.simx_opmode_oneshot)
                    time.sleep(.1)
                    continue
                adjacency_matrix[idx_from][idx_to] = 3
                idx_to_set.append(idx_to)
                sim.simxStopSimulation(clientID, sim.simx_opmode_oneshot)
                time.sleep(.1)
                sim.simxStartSimulation(clientID, sim.simx_opmode_oneshot)
                time.sleep(.1)

with open('Amendments/scipts for graph construction/graph construction 0522/05.adjacency_matrix_and_transition_states_tested.txt', 'wb') as f:
    pickle.dump(adjacency_matrix, f, protocol=2)
    pickle.dump(transition_states, f, protocol=2)
