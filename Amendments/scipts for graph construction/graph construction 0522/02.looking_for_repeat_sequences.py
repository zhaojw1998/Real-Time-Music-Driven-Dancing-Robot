import pickle
from tqdm import tqdm

with open('Amendments/scipts for graph construction/graph construction 0522/00.primitive_compact.txt', 'rb') as f:
    primitive_compact = pickle.load(f)

repeat_sequence = {}
duration = 4
for idx in range(len(primitive_compact)-duration):
    sequence = primitive_compact[idx: idx+duration]
    if not str(sequence) in repeat_sequence:
        store = 1
    else:
        store = repeat_sequence[str(sequence)]  #for the case where a repeat instance shows up more than one time in different places
    idx_rest = idx + duration
    while idx_rest <= len(primitive_compact)-duration:
        sequence_sample = primitive_compact[idx_rest: idx_rest+duration]
        if not sequence_sample == sequence: #define a sequence as 'repeat' if it repeats iteslf several times sequentially
            if str(sequence) in repeat_sequence:
                idx_rest += (repeat_sequence[str(sequence)]-store)*duration
                repeat_sequence[str(sequence)] = max(store, repeat_sequence[str(sequence)]-store+1)
            else:
                idx_rest += duration
            break
        if not str(sequence) in repeat_sequence:
            repeat_sequence[str(sequence)] = 1
        repeat_sequence[str(sequence)] += 1
        idx_rest += duration

#supress redundent sequences (supress [1,2,3], [3,4,1] and the like if [1,2,3,4] already exists)
repeat_sequences_redundant_supress = {}
for Key in list(repeat_sequence.keys()):
    continue_flag = 0
    has_extended = 0
    Key_restore = Key[1:-1].split(',')
    Key_restore = list(map(int, Key_restore))
    key_restore = [key[1:-1].split(',') for key in list(repeat_sequences_redundant_supress.keys())]
    key_restore = [list(map(int, key)) for key in key_restore]
    for key in key_restore:
        if len(Key_restore) % len(key) == 0 and set(Key_restore) == set(key):
            if not has_extended:
                Key_restore.extend(Key_restore)
                has_extended = 1
            if any([key == Key_restore[i:i+len(key)] for i in range(0,len(Key_restore)//2)]):
                continue_flag = 1
                break
    if continue_flag == 0:
        repeat_sequences_redundant_supress[Key] = repeat_sequence[Key]
print(repeat_sequences_redundant_supress)
with open('Amendments/scipts for graph construction/graph construction 0522/02.repeat_sequences_redundant_supress.txt', 'wb') as f:
    pickle.dump(repeat_sequences_redundant_supress, f)