import pickle
import numpy as np 

with open('Amendments/scipts for graph construction/graph construction 0522/03.stable_typical_sequence_redundant_supress.txt', 'rb') as f:
    stable_typical_sequence_redundant_supress = pickle.load(f)

#print(stable_typical_sequence_redundant_supress)
state_set = []
for key in stable_typical_sequence_redundant_supress:
    for state in key:
        if not state in state_set:
            state_set.append(state)
#print(state_set, len(state_set))
state_set = sorted(state_set)   #state_set = {3, 4, 5, 6, 28, 29, ...}
state_idx_dict={}   #state_idx_dict = {state 3: idx 0, state 4: idx 1, ...}
for idx, state in enumerate(state_set):
    state_idx_dict[state] = idx
#print(state_idx_dict)
incidence_matrix = np.empty((0, len(state_set)), dtype='<U10')
interface = {}
interface['layer'] = []
interface['entrance'] = []
for sequence in stable_typical_sequence_redundant_supress:
    explored_set = [sequence[0]]
    hyperedge = np.zeros(len(state_set), dtype='<U10')
    #interface.append(state_idx_dict[sequence[0]])
    interface['layer'].append(incidence_matrix.shape[0])
    interface['entrance'].append(state_idx_dict[sequence[0]])
    for idx in range(len(sequence)-1):
        if not sequence[idx+1] in explored_set:
            hyperedge[state_idx_dict[sequence[idx]]] = state_idx_dict[sequence[idx+1]]
            explored_set.append(sequence[idx+1])
        else:
            hyperedge[state_idx_dict[sequence[idx]]] = str(state_idx_dict[sequence[idx+1]])+'_next'
            incidence_matrix = np.vstack((incidence_matrix, hyperedge))
            explored_set = [sequence[idx+1]]
            hyperedge = np.zeros(len(state_set), dtype='<U10')
            #interface.append(state_idx_dict[sequence[idx+1]])
    hyperedge[state_idx_dict[sequence[-1]]] = 'done'
    incidence_matrix = np.vstack((incidence_matrix, hyperedge))
print(incidence_matrix)
print(interface)
print(state_idx_dict)

with open('./04.incidence_matrix_and_interface.txt', 'wb') as f:
    pickle.dump(incidence_matrix, f, protocol=2)
    pickle.dump(interface, f, protocol=2)
    pickle.dump(state_idx_dict, f, protocol=2)
    pickle.dump(state_set, f, protocol=2)


        

