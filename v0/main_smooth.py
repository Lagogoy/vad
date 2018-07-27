import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import convnet
import sklearn.metrics as metrics
import wave
from feats_io import read_feats


parser = argparse.ArgumentParser(description='Human Activities Detection')
parser.add_argument('--wav', metavar='DIR', default='/home/caidanwei/HAD/data/wav',
                    help='path to dataset')
parser.add_argument('--feats', metavar='DIR', default='/home/caidanwei/HAD/data/test_feats',
                    help='path to dataset')
parser.add_argument('--results', metavar='DIR', default='/home/caidanwei/HAD/results',
                    help='path to save predicted labels')
parser.add_argument('-b', '--batch-size', default=5120, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--resume', default='model_best.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')


# frame_shift should be same as what in get_feature.py
frame_shift = 0.01


def main():
    global args, best_eer
    args = parser.parse_args()

    ext_frames = 15

    # create model
    model = convnet.TinyConvNet(2)
    print(model)
    model = torch.nn.DataParallel(model).cuda()

    # resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            return
    else:
        print("No model file. Exit.")
        return

    cudnn.benchmark = True

    ## evaluation ##
    cm = np.zeros([2, 2])
    cms = np.zeros([2, 2])
    test_set = set([os.path.splitext(x)[0] for x in os.listdir(args.feats)])
    for file_name in test_set:
        # load test data
        feats_file = os.path.join(args.feats, file_name)
        eval_dataset = HADDataset(feats_file, ext_frames)
        eval_loader = torch.utils.data.DataLoader(eval_dataset,
                          batch_size=args.batch_size, shuffle=False,
                          num_workers=16, pin_memory=True)
        
        # get test resullt
        output, target = validate(eval_loader, model)
        cm_, pred = getCM(output, target)
        cms_, preds = getCM(output, target, is_smooth=True)
        cm += cm_
        cms += cms_

        wav = wave.open(os.path.join(args.wav, file_name) + '.wav', 'r')
        # save prediction result
        getresult(os.path.join(args.results, file_name) + '_original', wav, pred, frame_shift)
        # save smoothed prediction result
        getresult(os.path.join(args.results, file_name) + '_smoothed', wav, preds, frame_shift)
        
        print(file_name)
        print('* Acc {acc:.3f}\tUAR {uar:.3f}\tSmooth Acc {accs:.3f}\t Smooth UAR {uars:.3f}'
              .format(acc=ACC(cm_), uar=UAR(cm_), accs=ACC(cms_), uars=UAR(cms_)))
    
    print('All test data')
    print('* Acc {acc:.3f}\tUAR {uar:.3f}\tSmooth Acc {accs:.3f}\t Smooth UAR {uars:.3f}'
          .format(acc=ACC(cm), uar=UAR(cm), accs=ACC(cms), uars=UAR(cms)))    
    return

        
class HADDataset(Dataset):
    def __init__(self, file_name, ext_frames):
        feats = []
        labels = []
        
        with open(file_name + '.idx', 'r') as idx_file:
            for line in idx_file:
                (frame_idx, feat_dsptr) = line.split(' ')
                feat, label = read_feats(feat_dsptr)
                feats.append(feat)
                labels.append(label)
        
        self.feats = np.transpose(np.array(feats, dtype=np.float32), (1, 2, 0))
        self.num_frames = self.feats.shape[-1]
        self.label = np.array(labels)
        self.ext_frames = ext_frames
        print('%d samples in' % (self.num_frames), end=" ")

    def __getitem__(self, idx):
        frame_idx = [0 if i < 0 else i 
                     if i < self.num_frames else self.num_frames-1 
                     for i in range(idx-self.ext_frames, idx+self.ext_frames+1)]
        feats = np.array(self.feats[:, :, frame_idx], dtype=np.float32)

        return self.label[idx], feats

    def __len__(self):
        return self.num_frames


def validate(val_loader, model):
    total_output = torch.zeros((0, 2)).type(torch.cuda.FloatTensor)
    total_target = torch.zeros((0,)).type(torch.cuda.LongTensor)
    
    # switch to evaluate mode
    model.eval()

    for i, (key, feats) in enumerate(val_loader):
        key = key.cuda(async=True)
        input_var = torch.autograd.Variable(feats, volatile=True)
        target_var = torch.autograd.Variable(key, volatile=True)

        # compute output
        output = model(input_var)

        total_output = torch.cat((total_output, output.data))
        total_target = torch.cat((total_target, key))

    return total_output, total_target


# confusion matrix
def getCM(output, target, is_smooth=False):
    _, pred = output.topk(1, 1, True, True)
    
    if is_smooth:
        pred = smooth_pred(pred.cpu().numpy())
    else:
        pred = pred.t().cpu().numpy()[0]

    target = target.view(-1).cpu().numpy()
    cm = metrics.confusion_matrix(target, pred, np.array([0, 1]))

    return cm, pred


def UAR(cm):
    uar = (cm[0, 0] / (float(sum(cm[0,:])) + 1e-10) + cm[1, 1] / (float(sum(cm[1,:]))) + 1e-10) / 2.0 * 100.0
    return uar


def ACC(cm):
    acc = (cm[0, 0] + cm[1, 1]) / (float(sum(sum(cm))) + 1e-10) * 100
    return acc


def smooth_pred(total_pred):
    num_frames = total_pred.shape[0]

    # speech segs should be at least 30 frames
    win_size = 30
    # remove short speech segments less than [win_size] frames
    silence_segs = frametags2segs(total_pred)
    smoothed_silence_segs = smooth_segs(num_frames, win_size, silence_segs)
    smoothed_pred = segs2frametags(num_frames, smoothed_silence_segs)

    # silence segs should be at least 100 frames
    win_size = 100
    # flip the smoothed segments, make the silence and speech ones reversed
    smoothed_pred = smoothed_pred ^ 1
    # remove the short silence segments less than [win_size] frames
    speech_segs = frametags2segs(smoothed_pred)
    smoothed_speech_segs = smooth_segs(num_frames, win_size, speech_segs)
    smoothed_pred = segs2frametags(num_frames, smoothed_speech_segs)
    
    smoothed_pred = smoothed_pred ^ 1
    
    return smoothed_pred.numpy()


# Generate positions of "0" segments in a "0/1" sequence
# Example:
#     Input:  [0 0 0 0 0 0 1 0 0 0 1 1 1 1 1 0 0 0 1 0 0]
#     Output: [[0, 5], [7, 9], [15, 17], [19, 20]]
def frametags2segs(frame_tags):
    num_frames = frame_tags.shape[0]
    silence_segs = []
    silence = False
    for i in range(0, num_frames):
        if silence and frame_tags[i] == 0:
            silence_segs[-1][1] = i
        elif silence and frame_tags[i] == 1:
            silence = False
        elif not silence and frame_tags[i] == 0:
            silence_segs.append([i, i])
            silence = True
        else:
            pass

    return silence_segs


def segs2frametags(num_frames, silence_segs):
    frame_tags = torch.ones(num_frames).type(torch.LongTensor)
    for seg in silence_segs:
        frame_tags[seg[0]:seg[1]+1] = 0

    return frame_tags


def smooth_segs(num_frames, win_size, silence_segs):
    smoothed_silence_segs = []
    for i, seg in enumerate(silence_segs):
        if i == 0 and seg[0] < win_size:
            smoothed_silence_segs.append([0, seg[1]])
        elif not i == 0 and seg[0] - silence_segs[i-1][1] < win_size:
            preSeg = smoothed_silence_segs.pop()
            smoothed_silence_segs.append([preSeg[0], seg[1]])
        else:
            smoothed_silence_segs.append(seg)

        if i == len(silence_segs) - 1 and num_frames - seg[1] < win_size:
            smoothed_silence_segs[-1][1] = num_frames - 1

    return smoothed_silence_segs


def getresult(file_name, wav, pred, frame_shift):
    # wave file with only predicted speech segments
    speech_wav = wave.open(file_name + '.wav', 'w')
    frame_rate = wav.getframerate()
    speech_wav.setnchannels(wav.getnchannels())
    speech_wav.setsampwidth(wav.getsampwidth())
    speech_wav.setframerate(frame_rate)
    
    # predicted speech & silence segments in seconds
    f = open(file_name + '.txt', 'w')
    
    segs = frametags2segs(pred)
    end = 0
    for seg in segs:
        start = seg[0] * frame_shift
        if end != start:
            f.write('%.3f\t%.3f\t%s\n' % (end, start, 'speech'))
            # write speech segment into wave file
            wav.setpos(int(end) * frame_rate)
            speech_wav.writeframesraw(wav.readframes(int(start - end) * frame_rate))
            
        end = seg[1] * frame_shift
        f.write('%.3f\t%.3f\t%s\n' % (start, end, 'silence'))
        
    f.close()
    speech_wav.close()


if __name__ == '__main__':
    main()
