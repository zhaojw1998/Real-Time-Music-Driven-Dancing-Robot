import pickle
import numpy as np 

with open('Amendments/scipts for graph construction/graph construction 0522/00.primitive_compact.txt', 'rb') as f:
    primitive_compact = pickle.load(f)
#print(primitive_compact)
with open('Amendments/scipts for graph construction/graph construction 0522/03.stable_typical_sequences_symmetrical.txt', 'rb') as f:
    stable_typical_sequences = pickle.load(f)
#print(stable_typical_sequence_redundant_supress, len(stable_typical_sequence_redundant_supress))
with open('Amendments/scipts for graph construction/graph construction 0522/04.nx5_matrix_and_interface.txt', 'rb') as f:
    nx5_matrix = pickle.load(f)
    interface = pickle.load(f)
with open('Amendments/scipts for graph construction/graph construction 0522/03.symmetric_reference.txt', 'rb') as f:
    symmetric_reference = pickle.load(f)
#print(incidence_matrix, interface)
states_in_typical_sequence = []
for sequence in stable_typical_sequences:
    for state in sequence:
        if state > 0:
            if not state in states_in_typical_sequence:
                states_in_typical_sequence.append(state)
        else:
            for original_state in symmetric_reference[state]:
                if not original_state in states_in_typical_sequence:
                    states_in_typical_sequence.append(original_state)
sequence_head = []
for sequence in stable_typical_sequences:
    sequence_head.append(sequence[0])
#print(states_in_typical_sequence, len(states_in_typical_sequence), sequence_head, len(sequence_head))
sequence_tail = []
for sequence in stable_typical_sequences:
    sequence_tail.append(sequence[-1])
    
transition_states = []
for state in list(set(primitive_compact + [state for state in sequence_tail if state < 0])):
    if state not in states_in_typical_sequence or state in sequence_head or state in sequence_tail:
        transition_states.append(state)
#print(sequence_head, sequence_tail)
transition_state_idx_dict = {}
for idx, state in enumerate(transition_states):
    transition_state_idx_dict[state] = idx
#print(transition_state_idx_dict)
print(transition_states, len(transition_states))
adjacency_matrix = np.zeros((len(transition_states), len(transition_states)))
for i in range(adjacency_matrix.shape[0]):
    if transition_states[i] > 0:
        next_options = [primitive_compact[(j+1)%len(primitive_compact)] for j in range(len(primitive_compact)) if primitive_compact[j] == transition_states[i]]
        for option in next_options:
            if (option in transition_state_idx_dict) and (option not in sequence_tail):
                adjacency_matrix[i][transition_state_idx_dict[option]] = 1
    if transition_states[i] < 0:
        for original_state in symmetric_reference[transition_states[i]]:
            next_options = [primitive_compact[(j+1)%len(primitive_compact)] for j in range(len(primitive_compact)) if primitive_compact[j] == original_state]
            for option in next_options:
                if (option in transition_state_idx_dict) and (option not in sequence_tail):
                    adjacency_matrix[i][transition_state_idx_dict[option]] = 1
#print(np.sum(adjacency_matrix, axis=1))
for state in sequence_head:
    for i in range(adjacency_matrix.shape[0]):
        if not (transition_states[i] in sequence_head or transition_states[i] in sequence_tail):
            #if not adjacency_matrix[i, transition_state_idx_dict[state]] == 1:
            adjacency_matrix[i, transition_state_idx_dict[state]] = 2
#print(np.sum(adjacency_matrix, axis=1))
for state in sequence_tail:
    for j in range(adjacency_matrix.shape[1]):
        if not (transition_states[j] in sequence_head or transition_states[j] in sequence_tail):
            if not adjacency_matrix[transition_state_idx_dict[state], j] == 1:
                adjacency_matrix[transition_state_idx_dict[state], j] = 2

#print(np.sum(adjacency_matrix, axis=1))
print(adjacency_matrix[transition_state_idx_dict[329]][transition_state_idx_dict[107]])
print(adjacency_matrix.shape)
#print(sequence_head)
#print(sequence_tail)
count = 0
for i in range(adjacency_matrix.shape[0]):
    if not transition_states[i] in sequence_head:
        if np.sum(adjacency_matrix[i] == 1)<1:
            count += 1
print(count)
with open('Amendments/scipts for graph construction/graph construction 0522/05.adjacency_matrix_and_transition_states.txt', 'wb') as f:
    pickle.dump(adjacency_matrix, f, protocol=2)
    pickle.dump(transition_states, f, protocol=2)
