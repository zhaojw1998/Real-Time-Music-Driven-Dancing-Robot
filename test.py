import torch
import numpy as np
from torch.nn import CosineSimilarity as CosineSimilarity
from torch.nn import LogSoftmax as LogSoftmax
from torch.nn import Softmax as Softmax

def loss_function(origin, target, random_1, random_2, random_3, random_4):
    cos = CosineSimilarity(dim=1, eps=1e-6)
    sim_1 = cos(origin, target).unsqueeze(1)    #batch_size * 1
    sim_2 = cos(origin, random_1).unsqueeze(1)
    sim_3 = cos(origin, random_2).unsqueeze(1)
    sim_4 = cos(origin, random_3).unsqueeze(1)
    sim_5 = cos(origin, random_4).unsqueeze(1)
    sim = torch.cat((sim_1, sim_2, sim_3, sim_4, sim_5), dim=1) #batch_size * compare_size
    logSoft = LogSoftmax(dim=1)
    output = torch.mean(logSoft(sim)[:, 0])
    return -output

cos = CosineSimilarity(dim=1, eps=1e-6)
logSoft = LogSoftmax(dim=1)
soft = Softmax(dim=1)

origin = torch.rand((15, 128))
target = torch.rand((15, 128))
random_1 = torch.rand((15, 128))
random_2 = torch.rand((15, 128))
random_3 = torch.rand((15, 128))
random_4 = torch.rand((15, 128))

sim_1 = cos(origin, target).unsqueeze(1)
sim_2 = cos(origin, random_1).unsqueeze(1)
sim_3 = cos(origin, random_2).unsqueeze(1)
sim_4 = cos(origin, random_3).unsqueeze(1)
sim_5 = cos(origin, random_4).unsqueeze(1)
print(sim_1.shape)

sim = torch.cat((sim_1, sim_2, sim_3, sim_4, sim_5), dim=1)
print(sim.shape)
print(soft(sim)[0, :])
print(logSoft(sim)[0, :])