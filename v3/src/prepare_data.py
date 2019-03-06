import argparse
import os
import librosa
import numpy as np 
from scipy import signal

parser = argparse.ArgumentParser(description="Extract Mfcc+delta+delta2 and labels")
parser.add_argument('--r', default = 5, type=int,
                    help='range of frames for labels (2*r+1 frames)')

args = parser.parse_args()
r = args.r
len_s = 0.025
shift_s = 0.01
sr = 8000
frame_len = int(len_s * sr)
frame_shift = int(shift_s * sr)

def cmvn(spec):
    ''' Cepstral Mean and Variance Normalization 
    '''
    mu = np.mean(spec, axis = 1)
    std = np.std(spec, axis = 1)
    return (spec - mu.reshape(-1,1))/std.reshape(-1,1)

def extract_features(file_path):
    ''' MFCC extraction
        sr = 8000, frame_len = 32ms, frame_shift = 16ms, mel_banks = 23, mfcc_dim = 12
    '''
    # 计算 MFCC
    y, _ = librosa.load(file_path, sr = sr)
    y = signal.lfilter([1, -0.97], 1, y)
    mfcc = librosa.feature.mfcc(
        y, sr=sr, n_fft=frame_len, hop_length=frame_shift, n_mels=23, n_mfcc=12
    )
    # 计算一阶和二阶差分
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order = 2)
    # 均值和标准差归一化
    mfcc = cmvn(mfcc)
    mfcc_delta = cmvn(mfcc_delta)
    mfcc_delta2 = cmvn(mfcc_delta2)
    return np.vstack((mfcc, mfcc_delta, mfcc_delta2)).T

def extract_label(filepath, featlen_dict):
    # 初始化labels
    label_dict = {}
    for wav in featlen_dict.keys():
        label_dict[wav] = np.zeros(featlen_dict[wav], dtype = np.float32)
    # 打开segment文件读取标签
    pfile = open(filepath, 'r')
    for line in pfile.readlines():
        temp = line.split()
        wav = temp[1]
        start = int(float(temp[2]) * sr / frame_shift)
        end   = int(float(temp[3]) * sr / frame_shift)
        label_dict[wav][start:end] = 1
    return label_dict


if __name__ == '__main__':
    wavfiles = os.listdir('../wav')
    wavfiles.sort()
    featlen_dict = {}
    os.system('mkdir feat label')
    # 计算MFCC和一阶二阶差分
    for wav in wavfiles:
        print(wav)
        feat = extract_features('../wav/' + wav)
        featlen_dict[wav.split('.')[0]] = feat.shape[0]
        np.save('feat/' + wav.split('.')[0], feat)
    # 计算label，文件源格式：RTTM
    label_dict = extract_label('../segment', featlen_dict)
    for (key, value) in label_dict.items():
        value = value.reshape(-1, 1)
        np.save('label/' + key.split('.')[0], value)
