import argparse
import os
import shutil
import time
import subprocess
from tensorboard_logger import configure, log_value
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import convnet
import sklearn.metrics as metrics
from feats_io import read_feats


parser = argparse.ArgumentParser(description='Human Activities Detection')
parser.add_argument('--data', metavar='DIR', default='/home/caidanwei/HAD/data/feats_ark',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=10240, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
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
parser.add_argument('--tensorboard', action='store_true',
                    help='use tensorboard to monitor the training process')
parser.add_argument('--name', default='', type=str,
                    help='directory to store tensorboard files')

best_prec = 0

def main():
    global args
    args = parser.parse_args()

    if args.tensorboard:
        if not args.name:
            raise RuntimeError('Please provide a name for tensorboard to store')
        configure("runs/{}".format(args.name))
        print('tensorboard is used, log to runs/{}'.format(args.name))
        
    # Data loading
    val_set = set(['1', '2', '1111_20170704_01', '1111_20170704_02',
                   'downlink1', 'uplink1', '17070401_01'])
    train_set = set([os.path.splitext(x)[0] for x in os.listdir(args.data)]) - val_set
    ext_frames = 15
    
    train_dataset = HADDataset(args.data, train_set, ext_frames)
    train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=None)

    eval_dataset = HADDataset(args.data, val_set, ext_frames)
    eval_loader = torch.utils.data.DataLoader(eval_dataset,
                  batch_size=args.batch_size, shuffle=False,
                  num_workers=args.workers, pin_memory=True)

    # create model
    model = convnet.TinyConvNet(2)
    print(model)

    model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

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
    def __init__(self, data_dir, data_set, ext_frames):
        feats_descriptor = {}
        num_frames = {}
        samples = []
        
        # read all *.idx file to get feature descriptor
        for file_name in data_set:
            feats_descriptor[file_name] = {}
            num_frames[file_name] = 0
            with open(os.path.join(data_dir, file_name + '.idx'), 'r') as idx_file:
                for line in idx_file:
                    (frame_idx, feat_dsptr) = line.split(' ')
                    feats_descriptor[file_name][int(frame_idx)] = feat_dsptr
                    num_frames[file_name] += 1
                samples = samples + [(file_name, i) for i in range(num_frames[file_name])]
        
        self.feats_descriptor = feats_descriptor
        self.num_frames = num_frames
        self.samples = samples
        self.ext_frames = ext_frames
        print('%d samples in all.' % len(self.samples))

    def __getitem__(self, index):
        file_name, cur_idx = self.samples[index]
        num_frames = self.num_frames[file_name]
        
        feats = []
        frame_idxs = [0 if i < 0 else i if i < num_frames else num_frames-1 
                     for i in range(cur_idx-self.ext_frames, cur_idx+self.ext_frames+1)]
        for frame_idx in frame_idxs:
            feat_dsptr = self.feats_descriptor[file_name][frame_idx]
            feat, _ = read_feats(feat_dsptr)
            feats.append(feat)

        feats = np.transpose(np.array(feats, dtype=np.float32), (1, 2, 0))
        _, label = read_feats(self.feats_descriptor[file_name][cur_idx])
        return label, feats

    def __len__(self):
        return len(self.samples)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    cm = np.zeros([2, 2], dtype=int)
    
    # switch to train mode
    model.train()

    end = time.time()
    for i, (key, feats) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        key = key.cuda(async=True)
        input_var = torch.autograd.Variable(feats)
        target_var = torch.autograd.Variable(key)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        losses.update(loss.data[0], feats.size(0))
        prec1, cm_ = accuracy(output.data, key)
        acc.update(prec1[0], feats.size(0))
        
        cm = cm + cm_
        uar_ = UAR(cm_)
        uar = UAR(cm)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {acc.val:.3f} ({acc.avg:.3f})\t'
                  'UAR {uar_:.3f} ({uar:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, acc=acc, uar_=uar_, uar=uar))
        
    if args.tensorboard:
        log_value('train_loss', losses.avg, epoch)
        log_value('train_acc', acc.avg, epoch)
        log_value('train_uar', uar, epoch)


def validate(val_loader, model, criterion, epoch=0):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    cm = np.zeros([2, 2], dtype=int)
    
    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (key, feats) in enumerate(val_loader):
        key = key.cuda(async=True)
        input_var = torch.autograd.Variable(feats, volatile=True)
        target_var = torch.autograd.Variable(key, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        losses.update(loss.data[0], feats.size(0))
        prec1, cm_ = accuracy(output.data, key)
        acc.update(prec1[0], feats.size(0))
        
        cm = cm + cm_
        uar_ = UAR(cm_)
        uar = UAR(cm)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print(' * Prec@1 {acc.avg:.3f}\tUAR {uar:.3f}'.format(acc=acc, uar=uar))
    
    if args.tensorboard:
        log_value('val_loss', losses.avg, epoch)
        log_value('val_acc', acc.avg, epoch)
        log_value('val_uar', uar, epoch)
    
    return acc.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


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
    lr = args.lr * (0.1 ** (epoch // 4))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    if args.tensorboard:
        log_value('lr', lr, epoch)


def accuracy(output, target):
    batch_size = target.size(0)

    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    # confusion matrix for UAR computation
    cm = metrics.confusion_matrix(target.view(1, -1).expand_as(pred).cpu().numpy()[0], pred.cpu().numpy()[0], np.array([0,1]))
    correct = correct.view(-1).float().sum(0, keepdim=True)
    acc = correct.mul_(100.0 / batch_size)
    return acc, cm


def UAR(cm):
    uar = (cm[0, 0] / (float(sum(cm[0,:])) + 1e-10) + cm[1, 1] / (float(sum(cm[1,:]))) + 1e-10) / 2.0 * 100.0
    return uar


if __name__ == '__main__':
    main()



