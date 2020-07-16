import numpy as np
import os
from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import fcluster
import json
import sys
import pickle

max_d = 7 #define the max distance between two clusters to be merged
#load dance primitives
dance_primitive_dir = 'danceprimitives'
dance_primitive = np.empty((0, 10*17*3))
primitives = os.listdir(dance_primitive_dir) # ['0000', '0001', '0002', ......]
for primitive in primitives:
    primitive_dir = os.path.join(dance_primitive_dir, primitive)
    primitive_norm = os.path.join(primitive_dir, 'dance_motion_normlized_' + str(int(primitive)) + '.npy')
    dance_primitive = np.vstack((dance_primitive, np.load(primitive_norm).reshape(-1)))
#cluster
Z = linkage(dance_primitive, 'average') #you may change measure
clusters = fcluster(Z, max_d, criterion='distance') #dimention = dance_primitive[0]. cluster[i] shows the cluster index which primitive i belongs to.
primitive_shrink = []
for i in range(len(clusters)):
    primitive_shrink.append(int(np.array(os.listdir(dance_primitive_dir))[np.array(clusters) == clusters[i]].tolist()[0]))
#print(primitive_shrink)
transition_baseline_1 = {}
for i in range(len(primitive_shrink)):
    if not primitive_shrink[i] in transition_baseline_1:
        next_state = [primitive_shrink[(j+1)%len(primitive_shrink)] for j in range(len(primitive_shrink)) if primitive_shrink[j] == primitive_shrink[i]]
        next_state = list(set(next_state))
        p = np.ones((len(next_state)))/len(next_state)
        transition_baseline_1[primitive_shrink[i]] = {}
        transition_baseline_1[primitive_shrink[i]]['next'] = next_state
        transition_baseline_1[primitive_shrink[i]]['p'] = p
        choice = np.random.choice(transition_baseline_1[primitive_shrink[i]]['next'], p=transition_baseline_1[primitive_shrink[i]]['p'])

#print(transition_baseline_1)

#with open('C:/Users/lenovo/Desktop/AI-Project-Portfolio/Amendments/experiment/baseline_graph.txt', 'wb') as f:
#    pickle.dump(transition_baseline_1, f)

if __name__ == '__main__':
    with open('C:/Users/lenovo/Desktop/AI-Project-Portfolio/Amendments/experiment/baseline_graph.txt', 'rb') as f:
        graph = pickle.load(f)
    #print(graph)
    state = 0
    record=[]
    while len(record) < 348:
        record.append(state)
        state = np.random.choice(graph[state]['next'], p=graph[state]['p'])
    print(record)
    with open('C:/Users/lenovo/Desktop/AI-Project-Portfolio/Amendments/experiment/record/baseline-record5.txt', 'wb') as f:
        pickle.dump(record, f, protocol=2)