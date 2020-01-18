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

primitive_transfer_graph = {}

for cluster_id in range(1, len(set(clusters))+1):
    primitive_transfer_graph[str(cluster_id)] = {}
    cluster_i = np.array(primitives)[np.array(clusters) == cluster_id].tolist()   #this should has dimention (n,) and n can be 1. CHECK!
    primitive_transfer_graph[str(cluster_id)]['include'] = cluster_i
    next_cluster_options = {}
    for primitive in cluster_i:
        next_cluster_id = clusters[(int(primitive) + 1)%len(set(clusters))]
        if not str(next_cluster_id) in next_cluster_options:
            next_cluster_options[str(next_cluster_id)] = 0
        next_cluster_options[str(next_cluster_id)] += 1/len(cluster_i)
    primitive_transfer_graph[str(cluster_id)]['lead_to'] = next_cluster_options #sorted(next_cluster_options.items(), key=lambda item:item[1], reverse=False)
#print(primitive_transfer_graph)
with open('primitive_transfer_graph.json', 'w') as f:
    json.dump(primitive_transfer_graph, f)
    
