import pickle
import numpy as np 

with open('Amendments/scipts for graph construction/graph construction 0522/03.stable_typical_sequences_symmetrical.txt', 'rb') as f:
    stable_typical_sequences = pickle.load(f)

#with open('Amendments/scipts for graph construction/graph construction 0522/03.stable_typical_sequences_asymmetrical.txt', 'rb') as f:
#    stable_typical_sequences = pickle.load(f) 

nx5_matrix = np.empty((0, 5), dtype=np.int16)
interface = []
for sequence in stable_typical_sequences:
    interface.append(nx5_matrix.shape[0])
    sequence_len = len(sequence)
    for i in range(sequence_len//4-1):
        nx5_matrix = np.vstack((nx5_matrix, np.array(sequence[4*i: 4*(i+1)]+[1])[np.newaxis, :]))
    nx5_matrix = np.vstack((nx5_matrix, np.array(sequence[sequence_len-4: sequence_len]+[-1])[np.newaxis, :]))
print(nx5_matrix)
print(nx5_matrix.shape)
print(interface)

#with open('Amendments/scipts for graph construction/graph construction 0522/04.nx5_matrix_and_interface.txt', 'wb') as f:
#    pickle.dump(nx5_matrix, f, protocol=2)
#    pickle.dump(interface, f, protocol=2)

#with open('Amendments/scipts for graph construction/graph construction 0522/04.nx5_matrix_and_interface_asymmetrical.txt', 'wb') as f:
#    pickle.dump(nx5_matrix, f, protocol=2)
#    pickle.dump(interface, f, protocol=2)