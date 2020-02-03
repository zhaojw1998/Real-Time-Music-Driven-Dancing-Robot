import numpy as np
import os
from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import fcluster
import sys
import pickle

#load dance primitives
dance_primitive_dir = 'danceprimitives'
#print(os.listdir(dance_primitive_dir))
dance_primitive = np.empty((0, 10*17*3))
for primitive in tqdm(os.listdir(dance_primitive_dir)):
    primitive_dir = os.path.join(dance_primitive_dir, primitive)
    primitive_norm = os.path.join(primitive_dir, 'dance_motion_normlized_' + str(int(primitive)) + '.npy')
    #print(np.load(primitive_norm).shape)
    dance_primitive = np.vstack((dance_primitive, np.load(primitive_norm).reshape(-1)))
#print(dance_primitive.shape)

#cluster
Z = linkage(dance_primitive, 'average') #you may change measure
c, coph_dists = cophenet(Z, pdist(dance_primitive))
print(c)

max_d = 7
clusters = fcluster(Z, max_d, criterion='distance') #this is a table containing primitive sequence where each item is the cluster index
#print(clusters)
#print(np.array(os.listdir(dance_primitive_dir))[np.array(clusters) == 1])
#print(type(np.array(os.listdir(dance_primitive_dir))[np.array(clusters) == 3]))
#print(np.array(os.listdir(dance_primitive_dir))[np.array(clusters) == 3].tolist()[0])
#print(len(clusters))
#print(min(clusters), max(clusters))
#print(len(set(clusters)))
print(clusters)
primitive_shrink = []
for i in range(len(clusters)):
    primitive_shrink.append(int(np.array(os.listdir(dance_primitive_dir))[np.array(clusters) == clusters[i]].tolist()[0]))
print(primitive_shrink)
assert(len(set(primitive_shrink)) == len(set(clusters)))
print(len(set(primitive_shrink)))

with open('C:\\Users\\lenovo\\Desktop\\AI-Project-Portfolio\\Amendments\\scipts for graph construction\\primitive_compact.txt', 'wb') as f:
    pickle.dump(primitive_shrink, f)
"""
plt.figure(figsize=(50, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(Z, leaf_rotation=90., leaf_font_size=8., color_threshold=7, show_leaf_counts=True, show_contracted=True)#, truncate_mode='lastp', p=len(set(clusters)))
plt.show()
"""