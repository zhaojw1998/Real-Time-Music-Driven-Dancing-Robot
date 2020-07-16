import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

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

if __name__ == '__main__':
    batchData = np.random.rand(2, 16, 17*32)
    model = Relevance(space_dims=17*32, hidden_dims=1024, relevance_dims=128).cuda()
    batchData = torch.from_numpy(batchData).float().cuda()
    output = model(batchData).detach().cpu().numpy()
    print(output.shape)