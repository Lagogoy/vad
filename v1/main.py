import argparse
import os
import shutil
import time
import subprocess
# from tensorboard_logger import configure, log_value
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import convnet
import sklearn.metrics as metrics
# from feats_io import read_feats


parser = argparse.ArgumentParser(description='Human Activities Detection')
parser.add_argument('--feats', default='data/mfcc.npy', type=str, 
                    help='the path of feats file')
parser.add_argument('--labels', default='data/labels.npy', type=str,
                    help='the path of labels file')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
# parser.add_argument('--tensorboard', action='store_true',
#                     help='use tensorboard to monitor the training process')
# parser.add_argument('--name', default='', type=str,
#                     help='directory to store tensorboard files')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    # if args.tensorboard:
    #     if not args.name:
    #         raise RuntimeError('Please provide a name for tensorboard to store')
    #     configure("runs/{}".format(args.name))
    #     print('tensorboard is used, log to runs/{}'.format(args.name))
    best_prec = 0
    ext_frame_num = 24
    train_percent = 0.9
 
    # Data loading
    feats = np.load(args.feats)
    labels = np.load(args.labels)
    train_frame_num = int(train_percent * len(labels))
    train_feats = feats[:,:,:train_frame_num]
    train_labels = labels[:train_frame_num]
    test_feats = feats[:,:,train_frame_num:]
    test_labels = labels[train_frame_num:]
    train_dataset = HADDataset(train_feats, train_labels, ext_frame_num)
    train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=12, pin_memory=True, sampler=None)

    eval_dataset = HADDataset(test_feats, test_labels, ext_frame_num)
    eval_loader = torch.utils.data.DataLoader(eval_dataset,
                  batch_size=args.batch_size, shuffle=False,
                  num_workers=12, pin_memory=True)

    # create model
    input_dim = [feats.shape[0], feats.shape[1], ext_frame_num*2+1]
    model = convnet.ConvNet(input_dim)
    print(model)
    model = torch.nn.DataParallel(model).to(device)

    # define loss function (criterion) and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    cudnn.benchmark = True

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec = checkpoint['best_prec']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.evaluate:
        validate(eval_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(eval_loader, model, criterion, epoch)

        # remember best eer and save checkpoint
        is_best = prec1 > best_prec
        best_prec = max(prec1, best_prec)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec': best_prec,
            'optimizer' : optimizer.state_dict(),
        }, is_best)


class HADDataset(Dataset):
    def __init__(self, feats, labels, ext_frame_num):
        self.feats = feats
        self.labels = labels 
        self.ext_frame_num = ext_frame_num

    def __getitem__(self, index):
        return_feats = self.feats[:, :, index:index+2*self.ext_frame_num+1]
        return_labels = self.labels[index+self.ext_frame_num].reshape(1)
        # feats = np.transpose(np.array(feats, dtype=np.float32), (1, 2, 0))
        return return_feats, return_labels

    def __len__(self):
        return len(self.labels) - 2*self.ext_frame_num


def train(train_loader, model, criterion, optimizer, epoch):
    losses = AverageMeter()
    acc = AverageMeter()
    cm = np.zeros([2, 2], dtype=int)

    model.train()
    for i, (feats, key) in enumerate(train_loader): 
        input_var = feats.float().to(device)
        target_var = key.float().to(device)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        losses.update(loss, feats.size(0))
        prec1, cm_ = accuracy(output, target_var)
        acc.update(prec1, feats.size(0))
        
        cm = cm + cm_
        uar_ = UAR(cm_)
        uar = UAR(cm)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % args.print_freq == 0:
            print('{} Epoch: [{}][{}/{}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Prec@1 {acc.val:.3f} ({acc.avg:.3f})\t'
                'UAR {uar_:.3f} ({uar:.3f})'.format(
                time.ctime(), epoch, i, len(train_loader), 
                loss=losses, acc=acc, uar_=uar_, uar=uar))

    # if args.tensorboard:
    #     log_value('train_loss', losses.avg, epoch)
    #     log_value('train_acc', acc.avg, epoch)
    #     log_value('train_uar', uar, epoch)
    return acc.avg


def validate(val_dataloader, model, criterion, epoch=0):
    losses = AverageMeter()
    acc = AverageMeter()
    cm = np.zeros([2, 2], dtype=int)

    model.eval()
    with torch.no_grad():
        for i, (feats, key) in enumerate(val_dataloader): 
            input_var = feats.float().to(device)
            target_var = key.float().to(device)

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            # measure accuracy and record loss
            losses.update(loss, feats.size(0))
            prec1, cm_ = accuracy(output, target_var)
            acc.update(prec1, feats.size(0))
            
            cm = cm + cm_
            uar_ = UAR(cm_)
            uar = UAR(cm)

            if i % args.print_freq == 0:
                print('{} Epoch: [{}][{}/{}]\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {acc.val:.3f} ({acc.avg:.3f})\t'
                    'UAR {uar_:.3f} ({uar:.3f})'.format(
                    time.ctime(), epoch, i, len(val_dataloader), 
                    loss=losses, acc=acc, uar_=uar_, uar=uar))

    # if args.tensorboard:
    #     log_value('train_loss', losses.avg, epoch)
    #     log_value('train_acc', acc.avg, epoch)
    #     log_value('train_uar', uar, epoch)
    return acc.avg


def save_checkpoint(state, is_best, filename='model/checkpoint.mdl'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model/model_best.mdl')


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


def adjust_learning_rate(optimizer, epoch):
    lr = args.lr * (0.1 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    # if args.tensorboard:
    #     log_value('lr', lr, epoch)


def accuracy(output, target):
    output = output.detach().cpu().numpy().squeeze()
    target = target.detach().cpu().numpy().squeeze()

    pred = (output >= 0.5).astype(int)
    target = target.astype(int)
    acc = (pred == target).sum() * 100.0 / len(target)
    # confusion matrix for UAR computation
    cm = metrics.confusion_matrix(target, pred, np.array([0,1]))
    return acc, cm


def UAR(cm):
    uar = ( cm[0,0]/(float(sum(cm[0,:])) + 1e-10) + cm[1,1]/(float(sum(cm[1,:])) + 1e-10) ) / 2.0
    return uar


if __name__ == '__main__':
    main()