import pickle
import numpy as np

with open('Amendments/scipts for graph construction/graph construction 0522/03.symmetric_reference.txt', 'rb') as f:
    symmetric_reference = pickle.load(f)
with open('Amendments/scipts for graph construction/graph construction 0522/00.primitive_compact.txt', 'rb') as f:
    primitive_compact = pickle.load(f)

with open('graph-record1.txt', 'rb') as f:
    record1 = pickle.load(f)
with open('graph-record2.txt', 'rb') as f:
    record2 = pickle.load(f)
with open('graph-record3.txt', 'rb') as f:
    record3 = pickle.load(f)

#printrecord()
sym_pair = []
for key in symmetric_reference:
    for item in symmetric_reference[key]:
        sym_pair.append([-int(key), item])
#print(sym_pair)
record_dict={}

for pair in sym_pair:
    if pair[0] <= 43:
        n_last = 0
        sym_record = []
        for n in range(len(record1)):
            if record1[n] == pair[1]:
                for m in range(n, -1, -1):
                    if record1[m] == pair[0]:
                        if m > n_last:
                            sym_record.append(n-m)
                        break
                n_last = n
        if not len(sym_record) == 0:
            if not str(pair) in record_dict:
                record_dict[str(pair)] = []
            record_dict[str(pair)] += sym_record

for pair in sym_pair:
    if pair[0] <= 43:
        n_last = 0
        sym_record = []
        for n in range(len(record2)):
            if record2[n] == pair[1]:
                for m in range(n, -1, -1):
                    if record2[m] == pair[0]:
                        if m > n_last:
                            sym_record.append(n-m)
                        break
                n_last = n
        if not len(sym_record) == 0:
            if not str(pair) in record_dict:
                record_dict[str(pair)] = []
            record_dict[str(pair)] += sym_record

for pair in sym_pair:
    if pair[0] <= 43:
        n_last = 0
        sym_record = []
        for n in range(len(record3)):
            if record3[n] == pair[1]:
                for m in range(n, -1, -1):
                    if record3[m] == pair[0]:
                        if m > n_last:
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
#print(record_analysis)
record_total = []
for key in record_dict:
    record_total += record_dict[key]
print(np.mean(record_total), np.std(record_total))