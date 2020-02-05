import pickle
import numpy as np 
import sys

with open('C:\\Users\\lenovo\\Desktop\\AI-Project-Portfolio\\Amendments\\scipts for graph construction\\00.primitive_compact.txt', 'rb') as f:
    primitive_compact = pickle.load(f)
with open('C:\\Users\\lenovo\\Desktop\\AI-Project-Portfolio\\Amendments\\scipts for graph construction\\03.stable_typical_sequence_redundant_supress.txt', 'rb') as f:
    stable_typical_sequence_redundant_supress = pickle.load(f)
sequence_head = []
sequence_tail = []
for sequence in stable_typical_sequence_redundant_supress:
    sequence_head.append(sequence[0])
    sequence_tail.append(sequence[-1])
with open('C:\\Users\\lenovo\\Desktop\\AI-Project-Portfolio\\Amendments\\scipts for graph construction\\05.adjacency_matrix_and_transition_states_tested.txt', 'rb') as f:
    adjacency_matrix = pickle.load(f)
    transition_states = pickle.load(f)
with open('C:\\Users\\lenovo\\Desktop\\AI-Project-Portfolio\\Amendments\\scipts for graph construction\\04.incidence_matrix_and_interface.txt', 'rb') as f:
    incidence_matrix = pickle.load(f)
    interface = pickle.load(f)
#print(transition_states, len(transition_states))
transition_graph = {}
for state_idx in range(adjacency_matrix.shape[0]):
    state = transition_states[state_idx]
    transition_graph[state] = {}
    next_state = {}
    total = 0
    shrink_ratio = 0.9
    #if state_idx == 6:
    #    print()
    for next_idx in [i+1 for i in range(len(primitive_compact)) if primitive_compact[i] == state]:
        #if transition_states[state_idx] == 6 and primitive_compact[next_idx] == 3:
        #    sys.exit()
        if primitive_compact[next_idx] in transition_states and adjacency_matrix[state_idx][transition_states.index(primitive_compact[next_idx])] == 1:
            if primitive_compact[next_idx] in sequence_head and sequence_tail[sequence_head.index(primitive_compact[next_idx])]==transition_states[state_idx]:
                continue
            if not primitive_compact[next_idx] in next_state:
                next_state[primitive_compact[next_idx]] = 0
            next_state[primitive_compact[next_idx]] += 1
            total += 1
    if np.sum(adjacency_matrix[state_idx]) > len(next_state):
        for option in next_state:
            next_state[option] = (next_state[option] / total) * shrink_ratio
        supplement = {}
        for supplement_option in [transition_states[j] for j in range(adjacency_matrix.shape[1]) if adjacency_matrix[state_idx][j] == 1]:
            if not supplement_option in next_state:
                supplement[supplement_option] = (1-shrink_ratio*(len(next_state)>0))/(np.sum(adjacency_matrix[state_idx])-len(next_state))
        #print(len(supplement), np.sum(adjacency_matrix[state_idx]), len(next_state))
        assert(len(supplement) == np.sum(adjacency_matrix[state_idx])-len(next_state))
        transition_graph[state]['next'] = list(next_state.keys()) + list(supplement.keys())
        print(transition_graph[state]['next'])
        transition_graph[state]['p'] = [next_state[key]for key in list(next_state.keys())] + [supplement[key] for key in list(supplement.keys())]
        #print(transition_graph[state]['p'])
        #assert(np.abs(1-np.sum(transition_graph[state]['p'])) <1e-5)
        #print(np.sum(transition_graph[state]['p']))
        next_step=np.random.choice(transition_graph[state]['next'], p=np.array(transition_graph[state]['p']).ravel())
        #print(next_step)
        #assert(np.sum(transition_graph[state]['p']) == 1)
    else:
        for option in next_state:
            next_state[option] = (next_state[option] / total)
        transition_graph[state]['next'] = list(next_state.keys())
        transition_graph[state]['p'] = [next_state[key]for key in list(next_state.keys())]
        #assert(sum(transition_graph[state]['p']) == 1)
print(transition_graph)
with open('C:\\Users\\lenovo\\Desktop\\AI-Project-Portfolio\\Amendments\\scipts for graph construction\\06.transition_graph.txt', 'wb') as f:
    pickle.dump(transition_graph, f)

    
    

