import pickle

with open('Amendments/scipts for graph construction/graph construction 0522/00.primitive_compact.txt', 'rb') as f:
    primitive_compact = pickle.load(f)

with open('graph-record1.txt', 'rb') as f:
    record = pickle.load(f)

#print(primitive_compact)
#print(record)

distance_record = 0
for i in range(len(record)-1):
    tmp = 10000
    s=record[i]
    s_=record[i+1]
    for m in range(len(primitive_compact)):
        if primitive_compact[m] == s:
            for n in range(len(primitive_compact)):
                if primitive_compact[n] == s_:
                    if abs(m-n) < abs(tmp):
                        tmp = m-n
    distance_record += abs(tmp)
distance_record /= (len(record)-1)
print(distance_record)