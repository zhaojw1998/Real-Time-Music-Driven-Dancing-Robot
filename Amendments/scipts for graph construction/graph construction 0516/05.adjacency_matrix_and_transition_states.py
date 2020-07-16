import pickle
import numpy as np 

with open('./00.primitive_compact.txt', 'rb') as f:
    primitive_compact = pickle.load(f)
#print(primitive_compact)
with open('./03.stable_typical_sequence_redundant_supress.txt', 'rb') as f:
    stable_typical_sequence_redundant_supress = pickle.load(f)
#print(stable_typical_sequence_redundant_supress, len(stable_typical_sequence_redundant_supress))
with open('./04.incidence_matrix_and_interface.txt', 'rb') as f:
    incidence_matrix = pickle.load(f)
    interface = pickle.load(f)
#print(incidence_matrix, interface)
states_in_typical_sequence = []
for sequence in stable_typical_sequence_redundant_supress:
    for state in sequence:
        if not state in states_in_typical_sequence:
            states_in_typical_sequence.append(state)
sequence_head = []
for sequence in stable_typical_sequence_redundant_supress:
    sequence_head.append(sequence[0])
#print(states_in_typical_sequence, len(states_in_typical_sequence), sequence_head, len(sequence_head))
sequence_tail = []
for sequence in stable_typical_sequence_redundant_supress:
    sequence_tail.append(sequence[-1])
    
transition_states = []
for state in list(set(primitive_compact)):
    if state not in states_in_typical_sequence or state in sequence_head or state in sequence_tail:
        transition_states.append(state)
#print(sequence_head, sequence_tail)
transition_state_idx_dict = {}
for idx, state in enumerate(transition_states):
    transition_state_idx_dict[state] = idx
#print(transition_state_idx_dict)
#print(transition_states, len(transition_states))
adjacency_matrix = np.zeros((len(transition_states), len(transition_states)))
for i in range(adjacency_matrix.shape[0]):
    next_options = [primitive_compact[(j+1)%len(primitive_compact)] for j in range(len(primitive_compact)) if primitive_compact[j] == transition_states[i]]
    for option in next_options:
        if option in transition_state_idx_dict:
            adjacency_matrix[i][transition_state_idx_dict[option]] = 1
#print(np.sum(adjacency_matrix, axis=1))
for state in sequence_head:
    adjacency_matrix[:, transition_state_idx_dict[state]] = 1
#print(np.sum(adjacency_matrix, axis=1))
for state in sequence_tail:
    adjacency_matrix[transition_state_idx_dict[state], :] = 1
#print(np.sum(adjacency_matrix, axis=1))
print(adjacency_matrix)
with open('./05.adjacency_matrix_and_transition_states.txt', 'wb') as f:
    pickle.dump(adjacency_matrix, f)
    pickle.dump(transition_states, f)
