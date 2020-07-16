import pickle
with open('Amendments/scipts for graph construction/graph construction 0522/04.nx5_matrix_and_interface.txt', 'rb') as f:
    nx5_matrix = pickle.load(f)
    interface = pickle.load(f)
print(nx5_matrix, interface)