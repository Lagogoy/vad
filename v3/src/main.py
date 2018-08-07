# pylint:disable=E1101, E1102
import argparse
import os
import time
import numpy as np 
import torch
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn 
import model
from tqdm import tqdm



parser = argparse.ArgumentParser(description="Speech Changepoint Detection")
parser.add_argument('--epochs', default=100, type=int,  
                    help='manual epoch number (default: 100)')
parser.add_argument('--lr', default=0.1, type=float,
                    help='initial learning rate')
parser.add_argument('--seg-len', default=3, type=int, 
                    help='segment length(s) as LSTM inputs')
parser.add_argument('--batch', default=64, type=int, 
                    help='mini batch')
parser.add_argument('--eval', action='store_true', 
                    help='whether evaluate model on validation set')
parser.add_argument('--resume', action='store_true', 
                    help='whether recovery from saved model')



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



# DataLoader类
class SCDDataset(Dataset):
    def __init__(self, feats, labels, seg_len):
        self.feats = feats
        self.labels = labels
        self.frame_num = seg_len * 100
        self.frame_shift = self.frame_num//2
    
    def __getitem__(self, index):
        feat  = self.feats[self.frame_shift*index : self.frame_shift*index + self.frame_num]
        label = self.labels[self.frame_shift*index : self.frame_shift*index + self.frame_num]
        return feat, label
    
    def __len__(self):
        return self.feats.shape[0]//(self.frame_shift) - 1



# 读取数据，生成train_loader和val_loader
def LoadData(data_set, percent):
    print("Start loading data...")
    feat_dict = {}; label_dict = {}
    for file in tqdm(data_set):
        feat = np.load('feat/' + file)
        label = np.load('label/' + file)
        feat_dict[file] = feat
        label_dict[file] = label
    
    # 训练集比例：data_set * percent
    # 测试集比例：data_set * (1 - percent)
    feats = np.vstack(feat_dict.values())
    labels = np.vstack(label_dict.values())
    train_feats  = feats[ : int(feats.shape[0]*percent)]
    train_labels = labels[ : int(feats.shape[0]*percent)]
    val_feats  = feats[int(feats.shape[0]*percent) : ]
    val_labels = labels[int(feats.shape[0]*percent) : ]

    seg_len = args.seg_len      # seg_len： 单位s，每次输入LSTM网络的长度
    train_dataset = SCDDataset(train_feats, train_labels, seg_len)
    train_loader  = torch.utils.data.DataLoader(train_dataset,
                    batch_size=args.batch, shuffle=True,
                    num_workers=12, pin_memory=True)
    val_dataset = SCDDataset(val_feats, val_labels, seg_len)
    val_loader  = torch.utils.data.DataLoader(val_dataset, 
                    batch_size=args.batch, shuffle=False,
                    num_workers=12, pin_memory=True)
    return train_loader, val_loader



def adjust_lr(optimizer, epoch):
    lr = args.lr * (0.1**(epoch//20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



def train(model, train_loader, criterion, optimizer, epoch):
    i = 1
    loss_static = AverageMeter()
    acc_static = AverageMeter()
    fa_static = AverageMeter()
    miss_static = AverageMeter()

    model.train()
    print('Train Model:')
    for feats, labels in tqdm(train_loader):
        feats = feats.transpose(0, 1).float().cuda()
        labels = labels.transpose(0, 1).float().cuda()
        model.init_hidden(batch = feats.shape[1])
        output = model(feats)
        loss = criterion(output, labels)
        loss_static.update(loss.item(), len(labels))

        acc, fa, miss = accuracy(output, labels)
        acc_static.update(acc, len(labels))
        fa_static.update(fa, len(labels))
        miss_static.update(miss, len(labels))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tqdm.write('{0} [TRAIN] Epoch:[{1}][{2}/{3}] Loss:{4:.4f}  Acc:{5:.4f} Fa:{6:.4f} Miss:{7:.4f}'.format(
            time.ctime(), epoch, i, len(train_loader), loss_static.avg, acc_static.avg, fa_static.avg, miss_static.avg
        ))
        i += 1



def val(model, val_loader, criterion, epoch):
    i = 1
    loss_static = AverageMeter()
    acc_static = AverageMeter()
    fa_static = AverageMeter()
    miss_static = AverageMeter()

    model.eval()
    print('Eval Model:')
    for feats, labels in tqdm(val_loader):
        feats = feats.transpose(0, 1).float().cuda()
        labels = labels.transpose(0, 1).float().cuda()
        model.init_hidden(batch = feats.shape[1])
        output = model(feats)
        loss = criterion(output, labels)
        loss_static.update(loss.item(), len(labels))

        acc, fa, miss = accuracy(output, labels)
        acc_static.update(acc, len(labels))
        fa_static.update(fa, len(labels))
        miss_static.update(miss, len(labels))

        tqdm.write('{0} [VAL] Epoch:[{1}][{2}/{3}] Loss:{4:.4f}  Acc:{5:.4f} Fa:{6:.4f} Miss:{7:.4f}'.format(
            time.ctime(), epoch, i, len(val_loader), loss_static.avg, acc_static.avg, fa_static.avg, miss_static.avg
        ))
        i += 1



def accuracy(output, target, threshold=0.5):
    output = output.reshape(1, -1).squeeze()
    target = target.reshape(1, -1).squeeze()
    length = len(target)

    pred = (output > threshold).float()
    TP = (pred+target==2).float().sum()
    TN = (pred+target==0).float().sum()
    FP = (pred-target ==1).float().sum()
    FN = (pred-target==-1).float().sum()

    acc = (TP + TN)/length
    fa  = FP/(FP + TP)
    miss = FN/(FN + TP)
    return acc, fa, miss
    


if __name__ == '__main__': 
    global args
    args = parser.parse_args()
    # 模型初始化，输入：12维MFCC + delta + delta2 = 36维，batch_size = 1
    model = model.lstmNet(input_dim = 36).cuda()
    print(model)

    # 读取数据
    data_set = os.listdir('feat')
    data_set.sort()
    train_percent = 0.9
    train_loader, val_loader = LoadData(data_set, train_percent)
    
    # 损失函数: 交叉熵
    criterion = nn.BCELoss().cuda()
    # 优化器：随机梯度下降法
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4
    )
    # 提升GPU运行速度
    cudnn.benchmark = True

    if(args.eval):
        args.resume = True

    start_epoch = 0
    # 读取存档
    if args.resume:
        assert(os.path.isfile('saved_point.bin'))
        saved_point = torch.load('saved_point.bin')
        start_epoch = saved_point['epoch'] + 1
        model.load_state_dict(saved_point['model'])
        optimizer.load_state_dict(saved_point['optimizer'])
        print("==> Load checkpoint: success! Restart from epoch:{}".format(start_epoch))
    
    if(args.eval):
        val(model, val_loader, criterion, start_epoch)
        exit(0)

    # 开始训练
    for epoch in range(start_epoch, args.epochs):
        adjust_lr(optimizer, epoch)
        train(model, train_loader, criterion, optimizer, epoch)
        val(model, val_loader, criterion, epoch)
        torch.save(
        {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }, 'saved_point.bin')
