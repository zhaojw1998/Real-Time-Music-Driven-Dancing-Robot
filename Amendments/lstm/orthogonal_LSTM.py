import torch
import torch.nn as nn

class orthogonalLSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layer):
        super(orthogonalLSTM, self).__init__()
        self.num_layer = num_layer
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layer)
        #self.linear = nn.Linear(hidden_dim*num_layer*2, 101)


    def forward(self, x):
        hn_0 = torch.randn(self.num_layer, 1, self.hidden_dim) #(num_layers,batch,output_size)
        cn_0 = torch.randn(self.num_layer, 1, self.hidden_dim) #(num_layers,batch,output_size)
        hn_0 = hn_0.cuda()
        cn_0 = cn_0.cuda()
        _, (hn, cn) = self.lstm(x, (hn_0, cn_0))
        #_, (hn_y, cn_y) = self.lstm(Fy, (hn_y, cn_y))
        #lstm_output = torch.cat((hn_x.view(1, -1), hn_y.view(1, -1)), 1).view(1, -1)
        #out = self.linear(lstm_output)
        return hn[0]

class inferencelLSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layer):
        super(inferencelLSTM, self).__init__()
        self.num_layer = num_layer
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layer)
        #self.linear = nn.Linear(hidden_dim*num_layer*2, 101)


    def forward(self, x, hn_0, cn_0):
        hn_0 = hn_0.cuda()
        cn_0 = cn_0.cuda()
        _, (hn, cn) = self.lstm(x, (hn_0, cn_0))
        #_, (hn_y, cn_y) = self.lstm(Fy, (hn_y, cn_y))
        #lstm_output = torch.cat((hn_x.view(1, -1), hn_y.view(1, -1)), 1).view(1, -1)
        #out = self.linear(lstm_output)
        return hn, cn