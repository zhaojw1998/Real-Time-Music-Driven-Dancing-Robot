import json
import torch
import os
import numpy as np
import time
import sys
from collections import OrderedDict
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss as CrossEntropy
from torch.optim.lr_scheduler import ExponentialLR
#from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset
from model import Capability_Naive, CapabilityWithAttention
from coordinate2angle import coordinate2angle
from manage_joints import get_first_handles
from transmit import transmit
import sim

class MinExponentialLR(ExponentialLR):
    def __init__(self, optimizer, gamma, minimum, last_epoch=-1):
        self.min = minimum
        super(MinExponentialLR, self).__init__(optimizer, gamma, last_epoch=last_epoch)

    def get_lr(self):
        return [
            max(base_lr * self.gamma**self.last_epoch, self.min)
            for base_lr in self.base_lrs
        ]

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

class unit_data_loader(Dataset):
    def __init__(self, primitive_data, ground_truth):
        self.primitives = np.load(primitive_data)
        self.ground_truth = np.load(ground_truth)
    
    def __getitem__(self, idx):
        return self.primitives[idx], self.ground_truth[idx]

    def __len__(self):
        return self.primitives.shape[0]

def get_label(unit_data, converter, clientID, Body):    #1 for stable and 0  for fall down
    labels = []
    continue_flag = 0
    duration = 0.03
    for i in tqdm(range(unit_data.shape[0])):
        sequence_sample = unit_data[i]
        #for duration in [0.025, 0.035]:
        for idx_f in range(sequence_sample.shape[0]):
            angle_recon = converter.frameRecon(sequence_sample[idx_f])
            # angles: LSP, LSR, LEY, LER, RSP, RSR, REY, RER, LHP, LHR, LHYP, LKP, RHP, RHR, RHYP, RKP, LAP, LAR, RAP, RAR
            angles = converter.generateWholeJoints(angle_recon)
            assert(len(angles)==20)
            transmit(clientID, Body, angles)
            time.sleep(duration)
            returnCode, position=sim.simxGetObjectPosition(clientID, Body['HeadYaw'], -1, sim.simx_opmode_buffer)
            if position[2] < 0.4 and position[2] > 0:   #fall down
                print('fall')
                sim.simxStopSimulation(clientID, sim.simx_opmode_oneshot)
                print('stop')
                time.sleep(.3)
                sim.simxStartSimulation(clientID, sim.simx_opmode_oneshot)
                print('start')
                time.sleep(.5)
                labels.append(0)
                continue_flag = 1
                break
        if continue_flag:
            continue_flag = 0
            continue
        labels.append(1)
        time.sleep(.3)
    labels = np.array(labels)
    print(labels.shape, np.sum(labels==0))
    np.save('unit_stability_labels.npy', np.array(labels))

def train(model, dataloader, epoch, loss_function, optimizer, writer, scheduler):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()
    steps = len(dataloader)

    model.train()
    for step, (data_sample, target) in enumerate(dataloader):
        data_sample = data_sample.float().cuda()
        target = target.long().cuda()
        data_time.update(time.time() - end)
        #print(data_sample.shape)
        predict = model(data_sample)
        loss = loss_function(predict, target)
        loss.backward()
        losses.update(loss.item())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()

        if (step + 1) % 1 == 0:
            print('-------------------------------------------------------')
            for param in optimizer.param_groups:
                print('lr1: ', param['lr'])
            print_string = 'Epoch: [{0}][{1}/{2}]'.format(epoch, step + 1, steps)
            print(print_string)
            print_string = 'data_time: {data_time:.3f}, batch time: {batch_time:.3f}'.format(
                data_time=data_time.val,
                batch_time=batch_time.val)
            print(print_string)
            print_string = 'loss: {loss:.5f}'.format(loss=losses.avg)
            print(print_string)
        writer.add_scalar('train/loss_total-epoch', losses.avg, epoch*steps+step)
    return losses.avg

if __name__ == '__main__':
    model = CapabilityWithAttention(time_dim=16)
    #model = Capability_Naive(space_dims=544, hidden_dims=1024, representation_dim=128)
    #model.load_state_dict(torch.load('./params-capability/Fri Jul 17 01-17-05 2020/best_fitted_params.pt')['model_state_dict'])
    
    run_time = time.asctime(time.localtime(time.time())).replace(':', '-')
    logdir = 'log-capability/' + run_time
    if not os.path.isdir(logdir):
        os.makedirs(logdir)
    save_dir = 'params-capability/' + run_time
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    writer = SummaryWriter(logdir)
    optimizer = optim.Adam(model.parameters(), 1e-3)
    scheduler = MinExponentialLR(optimizer, 0.99995, minimum=5e-6, last_epoch=-1)
    loss_function = CrossEntropy()  #weight=torch.Tensor([0.8, 0.2]).cuda()
    model.cuda()
    gpu_num = 1
    loss_record = 100
    # end of initialization

    for epoch in range(200):
        dataLoader = DataLoader(unit_data_loader('./dance_unit_data.npy', './unit_stability_labels.npy'), batch_size=32, drop_last=False, shuffle=True)
        loss = train(model, dataLoader, epoch, loss_function, optimizer, writer, scheduler)
        scheduler.step()
        if loss < loss_record:
            checkpoint = save_dir + '/best_fitted_params.pt'
            torch.save({'epoch': epoch, 'model_state_dict': model.cpu().state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, checkpoint)
            model.cuda()
            loss_record = loss


    """ip = '127.0.0.1'
    port = 19997
    sim.simxFinish(-1)  # just in case, close all opened connections
    clientID = sim.simxStart(ip, port, True, True, -5000, 5)
    # Connect to V-REP
    if clientID == -1:
        import sys
        sys.exit('\nV-REP remote API server connection failed (' + ip + ':' + str(port) + '). Is V-REP running?')
    print('Connected to Remote API Server')  # show in the terminal
    sim.simxStartSimulation(clientID, sim.simx_opmode_oneshot)
    Body = {}
    get_first_handles(clientID,Body)    
    errorCode, acc = sim.simxGetStringSignal(clientID, 'Acceleration', sim.simx_opmode_streaming)
    returnCode, position=sim.simxGetObjectPosition(clientID, Body['HeadYaw'], -1, sim.simx_opmode_streaming)

    converter = coordinate2angle()
    converter.set_bound(np.load('bound_range.npy'), np.load('step.npy'), 32, 16)
    unit_data = np.load('primitive_data.npy')

    get_label(unit_data, converter, clientID, Body)"""