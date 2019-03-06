import argparse
import os
import time
import numpy as np
from tensorboard_logger import configure, log_value

import torch 
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader

import convnet

parser = argparse.ArgumentParser(description='Voice Activity Detection')

parser.add_argument('--jobs', default=2, type=int,
                    help='number of data loading workers (default: 2)')
parser.add_argument('--epochs', default=100, type=int,  
                    help='manual epoch number (default: 100)')
parser.add_argument('--batch-size', default=256, type=int,
                    help='mini-batch size (default: 256)')
parser.add_argument('--lr', default=0.01, type=float,  
                    help='initial learning rate (default: 0.01)')
parser.add_argument('--print-freq', default=1, type=int, 
                    help='print frequency (default: 1)')
parser.add_argument('--resume', action='store_true', 
                    help='whether recovery from ./resume/checkpoint.pth.tar')
parser.add_argument('--evaluate', action='store_true', 
                    help='whether evaluate model on validation set')
parser.add_argument('--tensorboard', action='store_true',
                    help='whether use tensorboard to monitor the training process')



def main():

    best_prec = 0
    start_epoch = 0
    root_path = os.getcwd()
    model_path = os.path.join(root_path, 'best_model.pth')
    checkpoint_path = os.path.join(root_path, 'resume', 'checkpoint.pth.tar')    
    
    global args
    args = parser.parse_args()
    if args.tensorboard:
        configure("runs")
        print('tensorboard is used, cmd: tensorboard --logdir runs')

    merge_spec = np.load('merge_spec.npy')
    merge_labels = np.load('merge_labels.npy')
    print('speech percentage:{}/{}'.format(merge_labels.sum(), merge_labels.shape[0]))

    train_percent = 0.95
    train_num = int(merge_spec.shape[0] * train_percent)

    ext_frames = 40
    train_dataset = VADDataset(merge_spec[0:train_num, :], merge_labels[0:train_num], ext_frames)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                    batch_size=args.batch_size, shuffle=True, 
                    num_workers=args.jobs, pin_memory=True)
    eval_dataset = VADDataset(merge_spec[train_num:, :], merge_labels[train_num:], ext_frames)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, 
                    batch_size=args.batch_size, shuffle=False,
                    num_workers=args.jobs, pin_memory=True)

    model = convnet.ConvNet()
    print(model)
    model = torch.nn.DataParallel(model).cuda()

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    cudnn.benchmark = True

    if args.evaluate:
        if os.path.isfile(model_path):
            model.load_state_dict(torch.load(model_path))
            validate(eval_loader, model, criterion)
        else:
            print("No model found, EXIT")
        return

    if args.resume:
        if os.path.isfile(checkpoint_path):
            print("==> Loading checkpoint...")
            checkpoint = torch.load(checkpoint_path)
            start_epoch = checkpoint['epoch']
            best_prec = checkpoint['best_prec']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("==> Load checkpoint: success! Epoch: {}".format(start_epoch))
        else: 
            print("==> no checkpoint found")

    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        train(train_loader, model, criterion, optimizer, epoch)
        prec = validate(eval_loader, model, criterion)

        if(prec > best_prec):
            best_prec = prec
            torch.save(model.state_dict(), model_path)

        torch.save({'epoch': epoch+1,
                    'state_dict': model.state_dict(),
                    'best_prec': best_prec, 
                    'optimizer': optimizer.state_dict()
                    }, checkpoint_path)


class VADDataset(Dataset):
    def __init__(self, spec, labels, ext_frames):
        self.spec = spec
        self.labels = labels
        self.length = spec.shape[0]
        self.dim = spec.shape[1]
        self.ext_frames = ext_frames
        print('{} samples in all.'.format(self.length))

    def __getitem__(self, index):
        frames_idx = np.array(range(index-self.ext_frames, index+self.ext_frames+1))
        frames_idx[frames_idx > self.length - 1] = self.length - 1
        frames_idx[frames_idx < 0] = 0
        return self.spec[frames_idx], self.labels[index]

    def __len__(self):
        return self.length


def train(train_loader, model, criterion, optimizer, epoch):
    model.train()
    
    losses = AverageMeter()
    acc = AverageMeter()
    fa = AverageMeter()
    md = AverageMeter()
   
    for i, (feats, labels) in enumerate(train_loader):
        labels = labels.cuda(async=True)
        feats = feats.unsqueeze(1)
        input_var = torch.autograd.Variable(feats)
        target_var = torch.autograd.Variable(labels)
        
        output = model(input_var)
        loss = criterion(output, target_var)

        losses.update(loss.data[0])
        correct, false_alarm, miss_detect = accuracy(output.data, labels)
        acc.update(correct)
        fa.update(false_alarm)
        md.update(miss_detect)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i%args.print_freq == 0):
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Correct: {acc.val:.4f} ({acc.avg:.4f})\t'
                  'False Alarm: {fa.val:.4f} ({fa.avg:.4f})\t'
                  'Miss Detection: {md.val:.4f} ({md.avg:.4f})'.format(
                  epoch, i, len(train_loader), loss=losses, acc=acc, fa=fa, md=md))            

    if args.tensorboard:
        log_value('train_loss', loss.avg, epoch)
        log_value('train_acc', acc.avg, epoch)
        log_value('lr', lr, epoch)
        log_value('train_false_alarm', fa.avg, epoch)
        log_value('train_miss_detecion', md.avg, epoch)


def validate(eval_loader, model, criterion, epoch=-1):
    model.eval()
    
    losses = AverageMeter()
    acc = AverageMeter()
    fa = AverageMeter()
    md = AverageMeter()

    for i,(feats, labels) in enumerate(eval_loader):
        labels = labels.cuda(async=True)
        feats = feats.unsqueeze(1)
        input_var = torch.autograd.Variable(feats, volatile=True)
        target_var = torch.autograd.Variable(labels, volatile=True)

        output = model(input_var)
        loss = criterion(output, target_var)

        losses.update(loss.data[0])
        correct, false_alarm, miss_detect = accuracy(output.data, labels)
        acc.update(correct)
        fa.update(false_alarm)
        md.update(miss_detect)

        print('(eval)Correct: {acc.val:.4f} ({acc.avg:.4f})\t'
              'False Alarm: {fa.val:.4f} ({fa.avg:.4f})\t'
              'Miss Detection: {md.val:.4f} ({md.avg:.4f})'.format(
               acc=acc, fa=fa, md=md))

    if (args.tensorboard and epoch!=-1):
        log_value('val_loss', losses.avg, epoch)
        log_value('val_acc', acc.avg, epoch)
        log_value('val_false_alarm', fa.avg, epoch)
        log_value('val_miss_detection', md.avg, epoch)
    return acc.avg


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.val = val
        self.sum = self.sum + val
        self.count += 1
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    lr = args.lr * (0.5**(epoch//4))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target):
    batch_size = target.size(0)
    
    _, pred = output.topk(k=1, dim=1, largest=True)
    pred = torch.squeeze(pred)
    
    vector = pred - target    
    
    false_alarm = (vector==1).float().sum()/batch_size
    miss_detect = (vector==-1).float().sum()/batch_size
    correct = (vector==0).float().sum()/batch_size
    return correct, false_alarm, miss_detect


if __name__ == '__main__':
    main()
