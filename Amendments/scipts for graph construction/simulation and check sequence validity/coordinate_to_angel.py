import pickle
import numpy as np 
import os 
from angel_set_copy import set_angel
import json

with open('Amendments/scipts for graph construction/graph construction 0522/03.typical_sequences_symmetrical.txt', 'rb') as f:
    typical_sequences = pickle.load(f)
with open('Amendments/scipts for graph construction/graph construction 0522/05.adjacency_matrix_and_transition_states.txt', 'rb') as f:
    adjacency_matrix = pickle.load(f)
    transition_states = pickle.load(f)
save_root = 'primitive_base'
states = []
for sequence in typical_sequences:
    for state in sequence:
        if not state in states:
            states.append(state)
for state in transition_states:
    if not state in states:
        states.append(state)
print(states)
print(len(states))
primitive_base = {}
for state in states:
    primitive = np.load('C:\\Users\\lenovo\\Desktop\\AI-Project-Portfolio\\danceprimitives\\'+'{0}'.format(str(abs(state)).zfill(4))+'\\dance_motion_'+str(abs(state))+'.npy').reshape(-1, 17, 3)
    motion = {}
    for i in range(primitive.shape[0]):
        motion[str(i)] = {}
        for j in range(primitive.shape[1]):
            motion[str(i)][str(j)] = [-primitive[i][j][2], primitive[i][j][0], -primitive[i][j][1]]
    primitive_base[state] = {}
    for frame in range(len(motion)):
        primitive_base[state][frame]={}
        primitive_base[state][frame]['angle_info'] = set_angel(motion, frame, state)
        primitive_base[state][frame]['num_frame'] = len(motion)
        primitive_base[state][frame]['dist_backward'] = frame
        primitive_base[state][frame]['dist_forward'] = len(motion) - frame
save_name = os.path.join(save_root, 'dance_primitive_library.json')
with open(save_name, 'w') as f:
        json.dump(primitive_base, f)


