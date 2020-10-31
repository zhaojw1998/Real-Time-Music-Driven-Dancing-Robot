import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from transformer.module import Encoder


class Relevance(nn.Module):
    def __init__(self, space_dims, hidden_dims, relevance_dims):
        super(Relevance, self).__init__()
        self.gru = nn.GRU(space_dims, hidden_dims, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(hidden_dims * 2, relevance_dims)
   
    def forward(self, x):
        x = self.gru(x)[-1]   #(numLayer*numDirection)* batch* hidden_size
        x = x.transpose_(0, 1).contiguous()  #batch* (numLayer*numDirection)* hidden_size
        x = x.view(x.size(0), -1)   #batch* (numLayer*numDirection*hidden_size), where numLayer=1, numDirection=2
        relevance_vector = self.linear(x)  #batch* relevance_dims
        return relevance_vector

class Capability_Naive(nn.Module):
    def __init__(self, space_dims, hidden_dims, representation_dim):
        super(Capability_Naive, self).__init__()
        self.lstm = nn.LSTM(space_dims, hidden_dims, num_layers=1, batch_first=True)
        self.linear1 = nn.Linear(hidden_dims, representation_dim)
        self.linear2 = nn.Linear(representation_dim, 2)
   
    def forward(self, x):
        output, (hn, cn) = self.lstm(x)   #(numLayer*numDirection)* batch* hidden_size
        x = hn.transpose_(0, 1).contiguous()  #batch* (numLayer*numDirection)* hidden_size
        x = x.view(x.size(0), -1)   #batch* (numLayer*numDirection*hidden_size), where numLayer=1, numDirection=2
        x = self.linear1(x)  #batch* relevance_dims
        classification = self.linear2(x)
        return classification

#input x： batch * num_joint * time_resolution * space_resolution
#   first: (batch * num_joint) * space_temporal_embedding
#   then: batch * num_joint * space_temporal_embedding
#   eqvl: batch * num_word * d_model


class CapabilityWithAttention(nn.Module):
    def __init__(self, num_layers_LSTM=1, joint_dim=17, resolution_dim=32, time_dim=16, hidden_dim=128, 
                        n_layers_Trans=1, n_head=1, d_model=128, d_k=64, d_v=64, d_hid=128, 
                        dropout=0, n_position=17, min_dist=16, max_dist=16, relative_pe=False):    #d_hid == d_model
        super(CapabilityWithAttention, self).__init__()
        self.time_dim = time_dim
        self.joint_dim = joint_dim
        self.resolution_dim = resolution_dim
        self.gru = nn.GRU(resolution_dim, hidden_dim, num_layers=num_layers_LSTM, batch_first=True, bidirectional=False)
        self.lstm = nn.LSTM(resolution_dim, hidden_dim, num_layers=num_layers_LSTM, batch_first=True, bidirectional=False)
        self.jointAttention = Encoder(hidden_dim*num_layers_LSTM, n_layers_Trans, n_head, d_k, d_v, d_model, d_hid, 
                                        dropout, n_position, min_dist, max_dist, relative_pe)
        self.linear1 = nn.Linear(d_hid, 1)
        self.linear2 = nn.Linear(joint_dim, 2)
        self.conv2d_1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2d_2 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=2, padding=1)
        #self.pooling_2 = nn.MaxPool2d(kernel_size=4)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(256, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(-1, self.time_dim, self.joint_dim, self.resolution_dim)
        x = x.transpose_(1, 2).contiguous()
        n_batch, n_joint, res_time, res_space = x.shape
        x = x.view(-1, res_time, res_space) # (batch * num_joint) * res_time * res_space
        hn = self.gru(x)[-1]
        #output, (hn, cn) = self.lstm(x) #hn: #(numLayer*numDirection)* (batch * num_joint)* hidden_size
        x = hn.transpose_(0, 1).contiguous()    #(batch * num_joint)* (numLayer*numDirection)* hidden_size, where numLayer=1, numDirection=1
        x = x.view(n_batch, n_joint, -1)    #batch * num_joint * hidden_size 
        x, attn_weights = self.jointAttention(x)  #batch * num_joint * d_model, where d_model = hidden_size
        #x = self.linear1(x).view(n_batch, -1)
        #x = self.linear2(x)
        #x = self.sigmoid(x)
        out = self.relu(self.pooling(self.conv2d_1(attn_weights)))
        #print(out.shape)
        out = self.relu(self.pooling(self.conv2d_2(out)))
        out = out.view(out.shape[0], -1)
        x = self.linear(out)
        return x

if __name__ == '__main__':
    batchData = np.random.rand(2, 17, 16, 32)
    model = CapabilityWithAttention().cuda()
    batchData = torch.from_numpy(batchData).float().cuda()
    out = model(batchData)
    #attn_weights = attn_weights.detach().cpu().numpy()
    print(out.shape)