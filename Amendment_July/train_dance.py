import json
import torch
import os
import numpy as np
from model import Relevance
from dance_data_loader import danceUnitDataLodaer
from torch import optim
from torch.nn import functional as F
from torch.nn import CosineSimilarity as CosineSimilarity
from torch.nn import LogSoftmax as LogSoftmax
from torch.optim.lr_scheduler import ExponentialLR
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import time
from collections import OrderedDict
import sys

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

def loss_function(origin, target, random_1, random_2, random_3, random_4):
    cos = CosineSimilarity(dim=1, eps=1e-6)
    sim_1 = cos(origin, target).unsqueeze(1)    #batch_size * 1
    sim_2 = cos(origin, random_1).unsqueeze(1)
    sim_3 = cos(origin, random_2).unsqueeze(1)
    sim_4 = cos(origin, random_3).unsqueeze(1)
    sim_5 = cos(origin, random_4).unsqueeze(1)
    sim = torch.cat((sim_1, sim_2, sim_3, sim_4, sim_5), dim=1) #batch_size * compare_size
    logSoft = LogSoftmax(dim=1)
    output = torch.mean(-logSoft(sim)[:, 0])
    return output

def train(model, train_dataloader, epoch, loss_function, optimizer, writer, scheduler):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    model.train()
    for step, (origin, target, random_1, random_2, random_3, random_4) in enumerate(train_dataloader):
        data_time.update(time.time() - end)
        origin = origin.float().cuda()
        target = target.float().cuda()
        random_1 = random_1.float().cuda()
        random_2 = random_2.float().cuda()
        random_3 = random_3.float().cuda()
        random_4 = random_4.float().cuda()

        #with torch.no_grad():       
        
        optimizer.zero_grad()
        origin = model(origin)
        target = model(target)
        random_1 = model(random_1)
        random_2 = model(random_2)
        random_3 = model(random_3)
        random_4 = model(random_4)

        loss = loss_function(origin, target, random_1, random_2, random_3, random_4)
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
            print_string = 'Epoch: [{0}][{1}/{2}]'.format(epoch, step + 1, len(train_dataloader))
            print(print_string)
            print_string = 'data_time: {data_time:.3f}, batch time: {batch_time:.3f}'.format(
                data_time=data_time.val,
                batch_time=batch_time.val)
            print(print_string)
            print_string = 'loss: {loss:.5f}'.format(loss=losses.avg)
            print(print_string)
    writer.add_scalar('train/loss_total-epoch', losses.avg, epoch)
    return losses.avg


"""
def validation(model, val_dataloader, epoch, loss_function, writer, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_recon = AverageMeter()
    losses_kl = AverageMeter()
    end = time.time()

    model.eval()
    with torch.no_grad():
        for step, (batch, c) in enumerate(val_dataloader):
            data_time.update(time.time() - end)
            encode_tensor = batch.float()
            c = c.float()
            rhythm_target = np.expand_dims(batch[:, :, :-2].sum(-1), -1)    #batch* n_step* 1
            rhythm_target = np.concatenate((rhythm_target, batch[:, :, -2:]), -1)   #batch* n_step* 3
            rhythm_target = torch.from_numpy(rhythm_target).float()
            rhythm_target = rhythm_target.view(-1, rhythm_target.size(-1)).max(-1)[1]
            target_tensor = encode_tensor.view(-1, encode_tensor.size(-1)).max(-1)[1]
            if torch.cuda.is_available():
                encode_tensor = encode_tensor.cuda()
                target_tensor = target_tensor.cuda()
                rhythm_target = rhythm_target.cuda()
                c = c.cuda()

            recon, recon_rhythm, dis1m, dis1s, dis2m, dis2s = model(encode_tensor, c)
            distribution_1 = Normal(dis1m, dis1s)
            distribution_2 = Normal(dis2m, dis2s)
            loss, l_recon, l_kl = loss_function(recon, recon_rhythm, target_tensor, rhythm_target, distribution_1, distribution_2, beta=args['beta'])
            losses.update(loss.item())
            losses_recon.update(l_recon.item())
            losses_kl.update(l_kl.item())
            batch_time.update(time.time() - end)
            end = time.time()

            if (step + 1) % args['display'] == 0:
                print('----validation----')
                print_string = 'Epoch: [{0}][{1}/{2}]'.format(epoch, step + 1, len(val_dataloader))
                print(print_string)
                print_string = 'data_time: {data_time:.3f}, batch time: {batch_time:.3f}'.format(
                    data_time=data_time.val,
                    batch_time=batch_time.val)
                print(print_string)
                print_string = 'loss: {loss:.5f}'.format(loss=losses.avg)
                print(print_string)

    writer.add_scalar('val/loss_total-epoch', losses.avg, epoch)
    writer.add_scalar('val/loss_recon-epoch', losses_recon.avg, epoch)
    writer.add_scalar('val/loss_KL-epoch', losses_kl.avg, epoch)
    return losses.avg
"""

def main():
    # some initialization    
    model = Relevance(space_dims=17*32, hidden_dims=1024, relevance_dims=32)
    
    run_time = time.asctime(time.localtime(time.time())).replace(':', '-')
    logdir = 'Amendment_July/log/' + run_time
    if not os.path.isdir(logdir):
        os.makedirs(logdir)
    save_dir = 'Amendment_July/params/' + run_time
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    writer = SummaryWriter(logdir)
    optimizer = optim.Adam(model.parameters(), 1e-3)
    scheduler = MinExponentialLR(optimizer, 0.977, minimum=1e-5, last_epoch=-1)
    model.cuda()
    gpu_num = 1
    loss_record = 100
    # end of initialization

    for epoch in range(200):
        dataloader = DataLoader(danceUnitDataLodaer('Amendment_july/dance_unit_data.npy'), batch_size = 4, shuffle = True, num_workers = 4, drop_last = True)
        loss = train(model, dataloader, epoch, loss_function, optimizer, writer, scheduler)
        if loss < loss_record:
            checkpoint = save_dir + '/best_fitted_params.pt'
            torch.save({'epoch': epoch, 'model_state_dict': model.cpu().state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, checkpoint)
            model.cuda()
            loss_record = loss
        scheduler.step()

if __name__ == '__main__':
    main()
