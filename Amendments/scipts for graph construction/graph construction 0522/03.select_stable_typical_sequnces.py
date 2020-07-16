import pickle
from tqdm import tqdm

#with open('Amendments/scipts for graph construction/graph construction 0522/03.typical_sequences_symmetrical.txt', 'rb') as f:
#    typical_sequences = pickle.load(f)

with open('Amendments/scipts for graph construction/graph construction 0522/03.typical_sequences_asymmetrical.txt', 'rb') as f:
    typical_sequences = pickle.load(f)

with open('Amendments/scipts for graph construction/graph construction 0522/03.unstable_sequences.txt', 'rb') as f:
    unstable_sequences = pickle.load(f)

#typical_sequences = [key[1:-1].split(',') for key in typical_sequences.keys()]
typical_sequences = [list(map(int, key)) for key in typical_sequences]

stable_typical_sequences = []
for idx in range(len(typical_sequences)):
    if not idx in unstable_sequences:
        stable_typical_sequences.append(typical_sequences[idx])
print(stable_typical_sequences)
print(len(stable_typical_sequences))

#supress redundent sequences (supress [1,2,3], [3,4,1] and the like if [1,2,3,4] already exists)
"""
stable_typical_sequence_redundant_supress = []
for Key in stable_typical_sequences[::-1]:  #supress [1,2,3] if [1,2,3,4] already exists
    continue_flag = 0
    for key in stable_typical_sequence_redundant_supress:
        if any([Key == key[i:i+len(Key)] for i in range(len(key)-len(Key)+1)]):
            continue_flag = 1
            break
    if not continue_flag:
        stable_typical_sequence_redundant_supress.append(Key)
stable_typical_sequences = stable_typical_sequence_redundant_supress

stable_typical_sequence_redundant_supress = []
for Key in stable_typical_sequences[::-1]:  #supress [2,3,4,1] if [1,2,3,4] already exists
    continue_flag = 0
    for key in stable_typical_sequence_redundant_supress:
        if len(key) == len(Key):
            key_extend = key + key
            if any([Key == key_extend[i:i+len(Key)] for i in range(len(key))]):
                continue_flag = 1
                break
    if not continue_flag:
        stable_typical_sequence_redundant_supress.append(Key)

print(stable_typical_sequence_redundant_supress)
print(len(stable_typical_sequence_redundant_supress))
"""
#with open('Amendments/scipts for graph construction/graph construction 0522/03.stable_typical_sequences_symmetrical.txt', 'wb') as f:
#    pickle.dump(stable_typical_sequences, f, protocol=2)

with open('Amendments/scipts for graph construction/graph construction 0522/03.stable_typical_sequences_asymmetrical.txt', 'wb') as f:
    pickle.dump(stable_typical_sequences, f, protocol=2)