import torch
import time
from orthogonal_LSTM import inferencelLSTM
from collections import OrderedDict

checkpoint = torch.load('C:\\Users\\lenovo\\Desktop\\AI-Project-Portfolio\\Amendments\\lstm\\model\\model_0000\\LSTM-primitives_epoch-29.pth.tar', map_location=lambda storage, loc: storage)['model_state_dict']
new_dict = OrderedDict()
for key in checkpoint:
    new_dict['l'+key.strip('module.')] = checkpoint[key]

model = inferencelLSTM(348, 348, 10)
model.load_state_dict(new_dict)

hn_0 = torch.zeros((10, 1, 348)) #(num_layers,batch,output_size)
cn_0 = torch.zeros((10, 1, 348)) #(num_layers,batch,output_size)

hn = hn_0
cn = cn_0
model.cuda()
x = torch.zeros((1, 1, 348))
x[:, :, 10] = 1
x = x.cuda()

time1 = time.time()
print(x.argmax())
while True:
    hn, cn = model(x, hn, cn)
    x = hn[0].view(1, 1, -1)
    print(x.argmax())
    if time.time() - time1 > .1:
        break

