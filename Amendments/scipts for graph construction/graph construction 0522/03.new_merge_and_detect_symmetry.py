import pickle
import sys
import numpy as np 
with open('Amendments/scipts for graph construction/graph construction 0522/01.typical_sequences.txt', 'rb') as f:
    typical_sequences = pickle.load(f)

#print(len(typical_sequences))
typical_sequences = [key[1:-1].split(',') for key in typical_sequences.keys()]
typical_sequences = [list(map(int, key)) for key in typical_sequences]

with open('Amendments/scipts for graph construction/graph construction 0522/00.primitive_compact.txt', 'rb') as f:
    primitive_compact = pickle.load(f)

dance_primitive_dir = 'C:/Users/lenovo/Desktop/AI-Project-Portfolio/danceprimitives'


def detect_symmetric(sequence):
    assert(np.abs(np.log2(len(sequence)) - int(np.log2(len(sequence)))) < 1e-10)
    if len(sequence) <= 2:
        return sequence
    norm = 0
    for i in range(len(sequence)//2):
        primitive = np.load(dance_primitive_dir+'/{0}'.format(str(sequence[i]).zfill(4))+'/dance_motion_normlized_'+str(sequence[i])+'.npy').reshape(-1, 17, 3)
        primitive_mirror = primitive
        #print(primitive_mirror[:, :, 0])
        primitive_mirror[:, :, 0] = -primitive_mirror[:, :, 0]
        primitive_counterpoint = np.load(dance_primitive_dir+'/{0}'.format(str(sequence[i+len(sequence)//2]).zfill(4))+'/dance_motion_normlized_'+str(sequence[i+len(sequence)//2])+'.npy').reshape(-1, 17, 3)
        norm += np.mean(np.square(primitive_mirror - primitive_counterpoint))
    norm /= (len(sequence)//2)
    sequence = detect_symmetric(sequence[0: len(sequence)//2]) + sequence[len(sequence)//2:]
    if norm <= 1.40:
        for i in range(len(sequence)//2):
            sequence[i+len(sequence)//2] = -sequence[i]
    else:
        sequence = sequence[0: len(sequence)//2] + detect_symmetric(sequence[len(sequence)//2:])
    return sequence

def merge(sequences, true_order):
    merged = []
    to_be_poped = []
    for sequence_i in sequences:
        pop_flag = 0
        for sequence_j in sequences:
            if sequence_i == sequence_j:
                continue
            merged_sequence = sequence_i + sequence_j
            if any([true_order[i: i+len(merged_sequence)] == merged_sequence for i in range(len(true_order)-len(merged_sequence)+1)]):
                pop_flag = 1
                merged.append(merged_sequence)
                if not sequence_j in to_be_poped:
                    to_be_poped.append(sequence_j)
        if pop_flag:
            if not sequence_i in to_be_poped:
                to_be_poped.append(sequence_i)
    for sequence in sequences:
        if not sequence in to_be_poped:
            merged.append(sequence)
    return merged

def merge_reccurent(sequences, true_order):
    merged = merge(sequences, true_order)
    if merged == sequences:
        return merged
    else:
        return merge_reccurent(merged, true_order)


merged = merge_reccurent(typical_sequences, primitive_compact)
#print(merged)
#print(len(merged))
merged_redundent_suppress = []
for Key in merged:  #supress [2,3,4,1] if [1,2,3,4] already exists
    continue_flag = 0
    for key in merged_redundent_suppress:
        if len(key) == len(Key):
            key_extend = key + key
            if any([Key == key_extend[i:i+len(Key)] for i in range(len(key))]):
                continue_flag = 1
                break
    if not continue_flag:
        merged_redundent_suppress.append(Key)
#print(merged_redundent_suppress)
#print(len(merged_redundent_suppress))
merged_symmetric = []
for sequence in merged_redundent_suppress:
    symmetric = detect_symmetric(sequence)
    merged_symmetric.append(symmetric)
#print(merged_symmetric)
#print(len(merged_symmetric))

symmetric_reference = {}
for idx, sequence in enumerate(merged_symmetric):
    for i in range(len(sequence)):
        if sequence[i] < 0:
            if not sequence[i] in symmetric_reference:
                symmetric_reference[sequence[i]] = []
            if not merged_redundent_suppress[idx][i] in symmetric_reference[sequence[i]]:
                symmetric_reference[sequence[i]] = symmetric_reference[sequence[i]] + [merged_redundent_suppress[idx][i]]
print(symmetric_reference)
print(len(symmetric_reference))

#with open('Amendments/scipts for graph construction/graph construction 0522/03.typical_sequences_symmetrical.txt', 'wb') as f:
#    pickle.dump(merged_symmetric, f, protocol=2)

#with open('Amendments/scipts for graph construction/graph construction 0522/03.symmetric_reference.txt', 'wb') as f:
#    pickle.dump(symmetric_reference, f, protocol=2)

#with open('Amendments/scipts for graph construction/graph construction 0522/03.typical_sequences_asymmetrical.txt', 'wb') as f:
#    pickle.dump(merged_redundent_suppress, f, protocol=2)
        
        

