import pickle

with open('Amendments/scipts for graph construction/graph construction 0522/00.primitive_compact.txt', 'rb') as f:
    primitive_compact = pickle.load(f)
#print(len(primitive_compact), len(record))
with open('C:/Users/lenovo/Desktop/AI-Project-Portfolio/Amendments/experiment/record/graph-record5.txt', 'rb') as f:
    record1 = pickle.load(f)

with open('C:/Users/lenovo/Desktop/AI-Project-Portfolio/Amendments/experiment/record/baseline-record4.txt', 'rb') as f:
    record2 = pickle.load(f)

global_record_1 = 0
for i in range(len(record1)-2):
    local_record = 10000
    for m in range(len(primitive_compact)):
        if primitive_compact[m] == record1[i]:
            for n in range(len(primitive_compact)):
                if primitive_compact[n] == record1[i+2]:
                    if abs(m-n) < local_record:
                        local_record = abs(m-n)
    global_record_1 += local_record
global_record_2 = 0
for i in range(len(record2)-2):
    local_record = 10000
    for m in range(len(primitive_compact)):
        if primitive_compact[m] == record2[i]:
            for n in range(len(primitive_compact)):
                if primitive_compact[n] == record2[i+2]:
                    if abs(m-n) < local_record:
                        local_record = abs(m-n)
    global_record_2 += local_record
print(global_record_1, global_record_2)