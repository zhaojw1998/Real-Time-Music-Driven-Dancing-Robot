import pickle
from tqdm import tqdm
import sys
with open('./00.primitive_compact.txt', 'rb') as f:
    primitive_compact = pickle.load(f)

typical_sequences = {}
max_duration = 32
for duration in range(4, max_duration+1, 1):    #detecting typical sequences with different duration
    for idx in range(len(primitive_compact)-duration):
        sequence = primitive_compact[idx: idx+duration]
        key_restore = [key[1:-1].split(',') for key in list(typical_sequences.keys())]  #transform '[1, 2, 3, 4]' to ['1', '2', '3', '4']
        key_restore = [list(map(int, key)) for key in key_restore]   #transform ['1', '2', '3', '4'] to [1, 2, 3, 4]
        continue_flag = 0
        for key in key_restore:
            if all([key[i%len(key)] == sequence[i] for i in range(len(sequence))]):
                continue_flag = 1   #neglect [1, 2, 3, 4, 1] if [1, 2, 3, 4] already exists
                break

        if continue_flag:
            continue
        for idx_rest in range(idx+duration, len(primitive_compact)-duration):
            sequence_sample = primitive_compact[idx_rest: idx_rest+duration]
            if sequence == sequence_sample: #define a sequence as 'typical' if it shows up more than once
                if not str(sequence) in typical_sequences:
                    typical_sequences[str(sequence)] = 1#[[idx, idx+duration]]
                typical_sequences[str(sequence)] += 1 #.append([idx_rest, idx_rest+duration])

print(typical_sequences)
print(len(typical_sequences))
with open('./01.typical_sequences.txt', 'wb') as f:
    pickle.dump(typical_sequences, f)
"""
#supress redundent sequences (supress [1,2,3], [3,4,1] and the like if [1,2,3,4] already exists)
typical_sequence_repeat_supress = {}    
for Key in list(typical_sequences.keys())[::-1]:
    continue_flag = 0
    key_restore = [key[1:-1].split(',') for key in list(typical_sequence_repeat_supress.keys())]
    key_restore = [list(map(int, key)) for key in key_restore] #restore the form [1, 2, 3, 4]
    Key_restore = Key[1:-1].split(',')
    Key_restore = list(map(int, Key_restore))   #restore the form [1, 2, 3, 4]
    for key in key_restore:
        key.extend(key)
        if any([Key_restore == key[i:i+len(Key_restore)] for i in range(0,len(key))]):  
            continue_flag = 1   #supress [1,2,3], [3,4,1] if [1,2,3,4] already exists
            break
    if continue_flag == 0:
        typical_sequence_repeat_supress[Key] = typical_sequences[Key]
"""
#print(len(typical_sequence_repeat_supress))
#print(typical_sequence_repeat_supress)

