import os
import glob
import time
import torch
import numpy as np
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
#from AudioDataLoader import audioDataLoader
from DataLoader_LSTM import DataLoaderLSTM
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

#from dataloaders.dataset import VideoDataset

from orthogonal_LSTM import orthogonalLSTM
from torchsummary import summary


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

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def train(model, train_dataloader, epoch, criterion, optimizer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()
    end = time.time()
    for step, file in enumerate(train_dataloader):
        file = file.permute(1, 0, 2)
        #print(file.shape)
        data_time.update(time.time() - end)
        x = file[:-1].cuda().float()
        #print(x.shape)
        y = file[-1].argmax().view(-1).cuda().long()
        #print(y)
        outputs = model(x).view(1, -1)
        #print(outputs.shape)
        loss = criterion(outputs, y)
        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs, y, topk=(1, 5))
        losses.update(loss.item(), 1)
        top1.update(prec1.item(), 1)
        top5.update(prec5.item(), 1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        if step % 100 == 0:
            print('-------------------------------------------------------')
            for param in optimizer.param_groups:
                print('lr1: ', param['lr'])
            print_string1 = 'Epoch: [{0}][{1}/{2}]'.format(epoch, step + 1, len(train_dataloader))
            print(print_string1)
            print_string2 = 'data_time: {data_time:.3f}, batch time: {batch_time:.3f}'.format(
                data_time=data_time.val,
                batch_time=batch_time.val)
            print(print_string2)
            print_string3 = 'loss: {loss:.5f}'.format(loss=losses.avg)
            print(print_string3)
            print_string4 = 'Top-1 accuracy: {top1_acc:.2f}%, Top-5 accuracy: {top5_acc:.2f}%'.format(
                top1_acc=top1.avg,
                top5_acc=top5.avg)
            print(print_string4)

def main():
    cudnn.benchmark = True
    cudnn.fastest = True
    cudnn.deterministic = False
    cudnn.enabled = True
    
    torch.cuda._lazy_init()
    writer = None#SummaryWriter(logdir)

    gpu_num = torch.cuda.device_count()
   
    

    print("Loading " + " dataset")

    train_dataloader = DataLoader(DataLoaderLSTM(),
                                  batch_size=1,
                                  shuffle=True,
                                  num_workers=4,
                                  drop_last=False)

    print("load model")
    model = orthogonalLSTM(348, 348, 10)
    #optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)#, weight_decay=5e-4)
    optimizer = optim.Adam(model.parameters(), 1e-2)#, weight_decay=5e-4)

    save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    # exp_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]
    
    #if params['resume_epoch'] != 0:
    #    models = sorted(glob.glob(os.path.join(save_dir_root, 'model', 'model_*')))
    #    model_id = int(models[-1].split('_')[-1]) if models else 0
    #else:
    models = sorted(glob.glob(os.path.join(save_dir_root, 'model', 'model_*')))
    model_id = int(models[-1].split('_')[-1]) + 1 if models else 0
    save_dir = os.path.join(save_dir_root, 'model', 'model_' + '%04d' % (model_id))
    saveName = 'LSTM' + '-' + 'primitives' #+ '-' + params['stream']
    
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    model = model.cuda()

    model = nn.DataParallel(model, device_ids=list(range(gpu_num)))  # multi-Gpu

    last_epoch = -1

    criterion = nn.CrossEntropyLoss().cuda()  # standard crossentropy loss for classification
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # the scheduler divides the lr by 10 every 10 epochs
    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=params['milestones'], gamma=0.1, last_epoch=last_epoch)
    
    for epoch in range(30):
        train(model, train_dataloader, epoch, criterion, optimizer)
        scheduler.step()
        last_epoch += 1  #CHANGE
        if (epoch+1) % 30 == 0:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            checkpoint = os.path.join(save_dir, saveName + '_epoch-' + str(epoch) + '.pth.tar')
            torch.save({'epoch':epoch,
                        'model_state_dict':model.state_dict(),
                        'optimizer_state_dict':optimizer.state_dict()},
                        checkpoint)
    



if __name__ == "__main__":
    main()