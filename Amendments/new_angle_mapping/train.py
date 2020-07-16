import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import numpy as np 
import os
import time
from tqdm import tqdm
from rodrigues_loss import RodriguesLoss

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

class coordDataSet(Dataset):
    def __init__(self, dataset): 
        self.dataset = dataset

    def __getitem__(self, index):
        coord = self.dataset[index]
        return coord

    def __len__(self):
        return self.dataset.shape[0]

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train(model, dataloader, epoch, criterion, optimizer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()
    end = time.time()
    for step, coord in enumerate(dataloader):
        data_time.update(time.time() - end)
        coord = coord.float()
        angle = model(coord)
        loss = criterion(angle, coord)
        # measure accuracy and record loss
        losses.update(loss.item(), coord.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()
        print('-------------------------------------------------------')
        for param in optimizer.param_groups:
            print('lr1: ', param['lr'])
        print_string1 = 'Epoch: [{0}][{1}/{2}]'.format(epoch, step + 1, len(dataloader))
        print(print_string1)
        print_string2 = 'data_time: {data_time:.3f}, batch time: {batch_time:.3f}'.format(
            data_time=data_time.val,
            batch_time=batch_time.val)
        print(print_string2)
        print_string3 = 'loss: {loss:.5f}'.format(loss=losses.avg)
        print(print_string3)
        i = np.random.randint(0, angle.shape[0])
        print('results glance:', angle[i])

def main():
    print('loading data')
    root = 'C:\\Users\\lenovo\\Desktop\\AI-Project-Portfolio\\danceprimitives'
    coord_data = np.empty((0, 51))
    for danceprimitive in tqdm(os.listdir(root)):
        primitive_dir = os.path.join(root, danceprimitive)
        primitive = np.load(os.path.join(primitive_dir, 'dance_motion_'+str(int(danceprimitive))+'.npy'))
        coord_data = np.vstack((coord_data, primitive))
    coord_data = torch.from_numpy(coord_data)
    dataloader = DataLoader(coordDataSet(coord_data), batch_size=128, shuffle=True, num_workers=4, drop_last=False)

    print('loading model')
    model = NeuralNetwork()
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    #optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)#, weight_decay=5e-4)
    criterion = RodriguesLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)  # the scheduler divides the lr by 10 every 10 epochs
    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=params['milestones'], gamma=0.1, last_epoch=-1)

    print('training')
    for epoch in range(400):
        train(model, dataloader, epoch, criterion, optimizer)
        scheduler.step()
    torch.save(model.state_dict(), 'C:\\Users\\lenovo\\Desktop\\AI-Project-Portfolio\\Amendments\\new_angle_mapping\\weights_2.pth')

if __name__ == "__main__":
    main()


