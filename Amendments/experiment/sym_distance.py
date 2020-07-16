import pickle
import numpy as np

with open('Amendments/scipts for graph construction/graph construction 0522/03.symmetric_reference.txt', 'rb') as f:
    symmetric_reference = pickle.load(f)
with open('Amendments/scipts for graph construction/graph construction 0522/00.primitive_compact.txt', 'rb') as f:
    primitive_compact = pickle.load(f)

sym_pair = []
for key in symmetric_reference:
    for item in symmetric_reference[key]:
        if not [-int(key), item] in sym_pair:
            sym_pair.append([-int(key), item])

record_dict={}
for i in range(1, 6):
    with open('C:/Users/lenovo/Desktop/AI-Project-Portfolio/Amendments/experiment/record/baseline-record'+str(i)+'.txt', 'rb') as f:
        record = pickle.load(f)
        print(len(set(record)))
    for pair in sym_pair:
        n_last = 0
        sym_record = []
        for n in range(len(record)):
            if record[n] == pair[1]:
                for m in range(n, -1, -1):
                    if record[m] == pair[0] and m > n_last:
                        sym_record.append(n-m)
                        break
                n_last = n
        if not len(sym_record) == 0:
            if not str(pair) in record_dict:
                record_dict[str(pair)] = []
            record_dict[str(pair)] += sym_record
            
#print(record_dict)
record_analysis = {}
for key in record_dict:
    record_analysis[key] = [np.mean(record_dict[key]), np.std(record_dict[key])]
print(record_analysis)
record_total = []
for key in record_dict:
    record_total += record_dict[key]
print(np.mean(record_total), np.std(record_total))