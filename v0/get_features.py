#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import librosa
import numpy as np
from scipy import signal
from textgrid import *
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
    y, sr = librosa.load(file_path, sr=None)
    y = signal.lfilter([1, -0.97], 1, y)
    spec = librosa.stft(y, n_fft=int(frame_len*sr), hop_length=int(frame_shift*sr), center=False)
    spec = librosa.feature.melspectrogram(S=np.abs(spec)**2+1e-10, n_mels=32)
    spec = cmvn(np.log(spec + 10e-10))
    spec_delta = cmvn(librosa.feature.delta(spec))
    spec_delta2 = cmvn(librosa.feature.delta(spec, order=2))
    spec = np.concatenate([np.array([spec]), np.array([spec_delta]), 
                           np.array([spec_delta2])], axis=0)
    return spec, sr

def extract_label(num_frames, frame_lens, file_path):
    '''
    extract the tag of each frame from .TextGrid file
    input: num_frames -- total frames of corresponding feature
           frame_lens -- number of seconds between input feature frames
    '''
    frames_label = np.empty(num_frames, dtype=int)
    label = TextGrid.load(file_path)
    interval_tier = label.tiers[0]
    for i, entry in enumerate(interval_tier.simple_transcript):
        start = int(float(entry[0]) / frame_lens)
        end = int(float(entry[1]) / frame_lens)
        tag = 0 if entry[2].find('s') == -1 else 1
        for j in range(start, min(num_frames, end)):
            frames_label[j] = tag
    return frames_label

if __name__ == '__main__':
    wav_dir = '/home/caidanwei/HAD/data/wav_converted'
    label_dir = '/home/caidanwei/HAD/data/label'
    feats_dir = '/home/caidanwei/HAD/data/feats_ark'
    
    for file_name in os.listdir(wav_dir):
        name, ext = os.path.splitext(file_name)
        file_path = os.path.join(wav_dir, file_name)
        feature, sr = log_mel_fbanks_energy(file_path)

        num_frames = feature.shape[-1]
        label_path = os.path.join(label_dir, name + '.TextGrid')
        labels = extract_label(num_frames, frame_shift, label_path)
        
        # write features and labels in *.idx & *.ark file
        feats_file = write_feats(os.path.join(feats_dir, name))
        for i in range(0, num_frames):
            feats_file.write(str(i), feature[:, :, i], labels[i])
        del feats_file
            
        print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),
              name + " write success")