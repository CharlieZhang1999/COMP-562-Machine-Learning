#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 1 2020
@author: Charlie Zhang(djogem)
@assignment: COMP 562 final project
"""

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torchvision.transforms as transforms
from torchsummary import summary
import os
import random
import shutil
import time
import warnings
import Custom_Alexnet
from Custom_Alexnet import CustomAlexnet, CustomTrainDataset, CustomValDataset

def main():
    if torch.cuda.is_available():  
      dev = "cuda:0" 
    else:  
      dev = "cpu"
    
    
    model = CustomAlexnet(200)
    datadir = './tiny-imagenet-200'#modify this path to your tiny-imagenet-200 path
    model.to(dev)
    
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(dev)
    
    lr = 0.005
    
    momentum = 0.9
    weight_decay = 1e-4
    optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                momentum=momentum,
                                weight_decay=weight_decay)
    
    # Data loading code
    traindir = os.path.join(datadir, 'train')
    valdir = os.path.join(datadir, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
    
    print('=> using CustomTrainDataset')
    train_dataset = CustomTrainDataset(
        traindir,
        transforms.Compose([
            # transforms.RandomResizedCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    print('=> using CustomValDataset')
    val_dataset = CustomValDataset(
        valdir,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    
    # The following 3 lines are for verification of your model structure
    # They should output exactly the same lines as the picture in the instruction
    print('For 3x64x64 images, the model will generate outputs in the following shapes:')
    summary(model, (3, 64, 64))
    assert train_dataset.classes == val_dataset.classes
    
    # Modify batch size to see the difference
    batch_size = 64
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True)
    
    
    best_acc1=0
    epochs=100
    for epoch in range(epochs):
    
        # train for one epoch
        train_loss = train(train_loader, model, criterion, optimizer, epoch, dev)
    
        # evaluate on validation set
        acc1, acc5, val_loss = validate(val_loader, model, criterion, dev)
    
        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
    
        # Don't forget to include model_best.pth.tar in your report
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)

#%%
#Helper Functions
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')
        
#%%
#Train and validation functions


def train(train_loader, model, criterion, optimizer, epoch, dev):

    print_frequency = 10
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.to(dev)
        target = target.to(dev)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_frequency == 0:
            progress.display(i)
        
    return losses.avg


def validate(val_loader, model, criterion, dev):

    print_frequency = 10
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(dev)
            target = target.to(dev)            

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_frequency == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg


if __name__ == '__main__':
    main()