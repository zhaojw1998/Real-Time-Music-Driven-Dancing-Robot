import pickle
import numpy as np 
import sys

with open('Amendments/scipts for graph construction/graph construction 0522/00.primitive_compact.txt', 'rb') as f:
    primitive_compact = pickle.load(f)
with open('Amendments/scipts for graph construction/graph construction 0522/03.stable_typical_sequences_symmetrical.txt', 'rb') as f:
    stable_typical_sequences = pickle.load(f)
sequence_head = []
sequence_tail = []
for sequence in stable_typical_sequences:
    sequence_head.append(sequence[0])
    sequence_tail.append(sequence[-1])
with open('Amendments/scipts for graph construction/graph construction 0522/05.adjacency_matrix_and_transition_states_tested.txt', 'rb') as f:
    adjacency_matrix = pickle.load(f)
    transition_states = pickle.load(f)
with open('Amendments/scipts for graph construction/graph construction 0522/04.nx5_matrix_and_interface.txt', 'rb') as f:
    nx5_matrix = pickle.load(f)
    interface = pickle.load(f)

#print(transition_states, len(transition_states))
limited_transition_graph = {}
for state_idx in range(adjacency_matrix.shape[0]):
    state = transition_states[state_idx]
    limited_transition_graph[state] = {}
    next_state = {}
    total = 0
    for next_idx in [(i+1)%len(primitive_compact) for i in range(len(primitive_compact)) if primitive_compact[i] == state]:
        #if transition_states[state_idx] == 6 and primitive_compact[next_idx] == 3:
        #    sys.exit()
        if primitive_compact[next_idx] in transition_states and adjacency_matrix[state_idx][transition_states.index(primitive_compact[next_idx])] == 1:
            #if primitive_compact[next_idx] in sequence_head and sequence_tail[sequence_head.index(primitive_compact[next_idx])]==transition_states[state_idx]:
            #    continue
            if not primitive_compact[next_idx] in next_state:
                next_state[primitive_compact[next_idx]] = 0
            next_state[primitive_compact[next_idx]] += 1
            total += 1
    if len(next_state) == 0:
        complement_states = [transition_states[j] for j in range(adjacency_matrix.shape[1]) if adjacency_matrix[state_idx][j] == 3]
        total = len(complement_states)
        for option in complement_states:
            next_state[option] = 1
    limited_transition_graph[state]['next'] = list(next_state.keys())
    limited_transition_graph[state]['p'] = [next_state[key]/total for key in list(next_state.keys())]
    #print(limited_transition_graph[state]['next'], limited_transition_graph[state]['p'])
    if not (state in sequence_head or state in sequence_tail):
        next_step=np.random.choice(limited_transition_graph[state]['next'], p=np.array(limited_transition_graph[state]['p']).ravel())
        assert(next_step not in sequence_head and next_state not in sequence_tail)
print(limited_transition_graph)

inter_transition_graph = {}
for state_idx in range(adjacency_matrix.shape[0]):
    state = transition_states[state_idx]
    inter_transition_graph[state] = {}
    next_state = {}
    total = 0
    shrink_ratio = 0.5
    for next_idx in [(i+1)%len(primitive_compact) for i in range(len(primitive_compact)) if primitive_compact[i] == state]:
        if primitive_compact[next_idx] in transition_states and adjacency_matrix[state_idx][transition_states.index(primitive_compact[next_idx])] == 1:
            if not primitive_compact[next_idx] in next_state:
                next_state[primitive_compact[next_idx]] = 0
            next_state[primitive_compact[next_idx]] += 1
            total += 1
    for option in next_state:
        next_state[option] = (next_state[option] / total)
    #print(next_state)
    supplement = {}
    supplement_options = [transition_states[j] for j in range(adjacency_matrix.shape[1]) if adjacency_matrix[state_idx][j]==2]
    for option in supplement_options:
        supplement[option] = 1
    for option in supplement:
        supplement[option] = (supplement[option] / len(supplement))

    if not (state in sequence_head or state in sequence_tail):
        assert(len(next_state) == (np.sum(adjacency_matrix[state_idx]==1)))
    assert(len(supplement) == (np.sum(adjacency_matrix[state_idx]==2)))

    inter_transition_graph[state]['next'] = list(next_state.keys()) + list(supplement.keys())
    #print(inter_transition_graph[state]['next'])
    if state not in sequence_head:
        assert(len(inter_transition_graph[state]['next']) > 0)
    p1 = [next_state[key] for key in list(next_state.keys())]
    p2 = [supplement[key] for key in list(supplement.keys())]
    prb = [p * shrink_ratio for p in p1] + [p * (1-shrink_ratio) for p in p2]
    inter_transition_graph[state]['p'] = [p / (sum(p1)*shrink_ratio + sum(p2)*(1-shrink_ratio)) for p in prb]

    if not (state in sequence_head):
        next_step=np.random.choice(inter_transition_graph[state]['next'], p=np.array(inter_transition_graph[state]['p']).ravel())

print(inter_transition_graph)
with open('Amendments/scipts for graph construction/graph construction 0522/06.transition_graph.txt', 'wb') as f:
    pickle.dump(limited_transition_graph, f, protocol=2)
    pickle.dump(inter_transition_graph, f, protocol=2)
    
    

