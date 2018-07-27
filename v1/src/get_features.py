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
    y, sr = librosa.load(file_path, sr=8000)
    y = signal.lfilter([1, -0.97], 1, y)
    spec = librosa.stft(y, n_fft=int(frame_len*sr), hop_length=int(frame_shift*sr), center=False)
    spec = librosa.feature.melspectrogram(S=np.abs(spec)**2+1e-10, n_mels=32)
    spec = cmvn(np.log(spec + 10e-10))
    spec_delta = cmvn(librosa.feature.delta(spec))
    spec_delta2 = cmvn(librosa.feature.delta(spec, order=2))
    spec = np.concatenate([np.array([spec]), np.array([spec_delta]), 
                           np.array([spec_delta2])], axis=0)
    return spec, sr


def extract_label(num_frames, frame_lens, file_path, name):
    '''
    extract the tag of each frame from .TextGrid file
    input: num_frames -- total frames of corresponding feature
           frame_lens -- number of seconds between input feature frames

    CHANGED by lawlict on April 11
    '''
    frames_label = np.zeros(num_frames, dtype=int)
    pfile = open(file_path, 'r')
    for line in pfile.readlines():
        utt, reco, start, end = line.split()
        if(name != reco):
            continue
        start = int(float(start)/frame_lens)
        end = min(int(float(end)/frame_lens), num_frames)
        frames_label[start:end] = 1
    return frames_label


if __name__ == '__main__':
    wav_dir = '../wav'
    label_path = '../segment'
    feats_dir = 'ark'
    
    for file_name in os.listdir(wav_dir):
        name, ext = os.path.splitext(file_name)
        file_path = os.path.join(wav_dir, file_name)
        feature, sr = log_mel_fbanks_energy(file_path)

        num_frames = feature.shape[-1]
        labels = extract_label(num_frames, frame_shift, label_path, name)
        
        # write features and labels in *.idx & *.ark file
        feats_file = write_feats(os.path.join(feats_dir, name))
        for i in range(0, num_frames):
            feats_file.write(str(i), feature[:, :, i], labels[i])
        del feats_file
            
        print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),
              name + " write success")
