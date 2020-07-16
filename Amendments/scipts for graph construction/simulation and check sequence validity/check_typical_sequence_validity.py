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

"""with open('graph construction 0516/01.typical_sequences.txt', 'rb') as f:
    typical_sequences = pickle.load(f)
"""
with open('Amendments/scipts for graph construction/graph construction 0522/03.typical_sequences_symmetrical.txt', 'rb') as f:
    typical_sequences = pickle.load(f)

#typical_sequences = [key[1:-1].split(',') for key in typical_sequences.keys()]
typical_sequences = [list(map(int, key)) for key in typical_sequences]

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
unstable_sequences = []
for idx, sequence in enumerate(typical_sequences):
    print('sequence:', sequence)
    continue_flag = 0
    for primitive_id in sequence:
        #print(primitive_id)
        primitive = np.load('C:\\Users\\lenovo\\Desktop\\AI-Project-Portfolio\\danceprimitives\\'+'{0}'.format(str(abs(primitive_id)).zfill(4))+'\\dance_motion_'+str(abs(primitive_id))+'.npy')
        primitive = primitive.reshape(-1, 17, 3)
        motion = {}
        for i in range(primitive.shape[0]):
            motion[str(i)] = {}
            for j in range(primitive.shape[1]):
                #motion[str(i)][str(j)] = [primitive[i][j][0], primitive[i][j][2], primitive[i][j][1]]
                motion[str(i)][str(j)] = [-primitive[i][j][2], primitive[i][j][0], -primitive[i][j][1]]
        for frame in range(primitive.shape[0]):
            joint_actuate(clientID, Body, motion, frame, primitive_id)
            returnCode, position=sim.simxGetObjectPosition(clientID, Body['HeadYaw'], -1, sim.simx_opmode_buffer)
            time.sleep(0.03)
            #print(position)
            if position[2] < 0.2 and position[2] > 0:   #fall down
                print(position[2], 'fall down in primitive', primitive_id, 'frame', frame)
                unstable_sequences.append(idx)
                #init_joint(clientID, Body, motion, frame)
                #time.sleep(1)
                sim.simxStopSimulation(clientID, sim.simx_opmode_oneshot)
                time.sleep(1)
                continue_flag = 1
                break
        if continue_flag:
            break
    if continue_flag:
        sim.simxStartSimulation(clientID, sim.simx_opmode_oneshot)
        time.sleep(1)
        continue

for idx, sequence in enumerate(typical_sequences):
    print('sequence:', sequence)
    continue_flag = 0
    for primitive_id in sequence:
        #print(primitive_id)
        primitive = np.load('C:\\Users\\lenovo\\Desktop\\AI-Project-Portfolio\\danceprimitives\\'+'{0}'.format(str(abs(primitive_id)).zfill(4))+'\\dance_motion_'+str(abs(primitive_id))+'.npy')
        primitive = primitive.reshape(-1, 17, 3)
        motion = {}
        for i in range(primitive.shape[0]):
            motion[str(i)] = {}
            for j in range(primitive.shape[1]):
                #motion[str(i)][str(j)] = [primitive[i][j][0], primitive[i][j][2], primitive[i][j][1]]
                motion[str(i)][str(j)] = [-primitive[i][j][2], primitive[i][j][0], -primitive[i][j][1]]
        for frame in range(primitive.shape[0]):
            joint_actuate(clientID, Body, motion, frame, primitive_id)
            returnCode, position=sim.simxGetObjectPosition(clientID, Body['HeadYaw'], -1, sim.simx_opmode_buffer)
            time.sleep(0.06)
            #print(position)
            if position[2] < 0.2 and position[2] > 0:   #fall down
                print(position[2], 'fall down in primitive', primitive_id, 'frame', frame)
                unstable_sequences.append(idx)
                #init_joint(clientID, Body, motion, frame)
                #time.sleep(1)
                sim.simxStopSimulation(clientID, sim.simx_opmode_oneshot)
                time.sleep(1)
                continue_flag = 1
                break
        if continue_flag:
            break
    if continue_flag:
        sim.simxStartSimulation(clientID, sim.simx_opmode_oneshot)
        time.sleep(1)
        continue

print(unstable_sequences)
print(len(unstable_sequences))
with open('Amendments/scipts for graph construction/graph construction 0522/03.unstable_sequences.txt', 'wb') as f:
    pickle.dump(unstable_sequences, f, protocol=2)
