import torch
from torch import nn  

class NeuralNetwork(nn.Module):
    def __init__(self, dropout = 0):
        super().__init__()
        self.lin1 = nn.Linear(51, 1024)
        self.lin2 = nn.Linear(1024, 512)
        self.lin3 = nn.Linear(512, 256)
        self.lin4 = nn.Linear(256, 128)
        self.lin5 = nn.Linear(128, 64)
        self.lin6 = nn.Linear(64, 22)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.normalize = nn.LayerNorm(51)

    def forward(self, x):
        #print(x.shape)
        x= self.normalize(x)
        x = self.relu(self.lin1(x))
        x = self.relu(self.lin2(x))
        x = self.relu(self.lin3(x))
        x = self.relu(self.lin4(x))
        x = self.relu(self.lin5(x))
        #x = self.dropout(x)
        x = self.lin6(x)
        return x