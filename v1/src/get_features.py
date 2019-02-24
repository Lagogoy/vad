#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import librosa
import numpy as np
from scipy import signal
from feats_io import write_feats

frame_len = 0.025
frame_shift = 0.01

def cmvn(spectrogram):
    '''Cepstral Mean and Variance Normalization
    '''
    mu = np.mean(spectrogram, axis=1)
    stdev = np.std(spectrogram, axis=1)
    return (spectrogram - mu.reshape((-1, 1))) / stdev.reshape((-1, 1))


def log_mel_fbanks_energy(file_path):
    y, sr = librosa.core.load(file_path, sr=None, mono=True)
    y = signal.lfilter([1, -0.97], 1, y)
    spec = librosa.stft(y, n_fft=int(frame_len*sr), hop_length=int(frame_shift*sr), center=False)
    spec = librosa.feature.melspectrogram(S=np.abs(spec)**2+1e-10, n_mels=32)
    spec = cmvn(np.log(spec + 10e-10))
    spec_delta = cmvn(librosa.feature.delta(spec))
    spec_delta2 = cmvn(librosa.feature.delta(spec, order=2))
    spec = np.array([spec, spec_delta, spec_delta2])
    return spec, sr

# FS_P01_dev_001  0   6.25    7.63    S   manual  X   X   X   X   X   X
def extract_label(num_frames, frame_shift, seg_path):
    frames_label = np.zeros(num_frames, dtype=int)
    with open(seg_path) as rfile:
        for line in rfile.readlines():
            items = line.strip().split()
            start = int(float(items[2])/frame_shift)
            end = min(int(float(items[3])/frame_shift), num_frames)
            frames_label[start:end] = 1
    return frames_label


if __name__ == '__main__':
    wav_dir = '/NASdata/AudioData/english/fearless_steps_challenge/Data/Audio/Tracks/Dev/'
    seg_dir = '/NASdata/AudioData/english/fearless_steps_challenge/Data/Transcripts/SAD/Dev/'
    feats = []
    labels = []
    for file_name in os.listdir(wav_dir):
        name, ext = os.path.splitext(file_name)
        feat_path = os.path.join(wav_dir, file_name)
        feature, sr = log_mel_fbanks_energy(feat_path)
        feats.append(feature)

        num_frames = feature.shape[-1]
        seg_path = seg_dir + name + '.txt'
        label = extract_label(num_frames, frame_shift, seg_path)
        labels.append(label)

        print(time.ctime(), name + " write success")
    feats = np.concatenate(feats, axis=1)
    labels = np.concatenate(labels, axis=0)
    np.save('mfcc.npy', feats)
    np.save('labels.npy', labels)