#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np 
import librosa
import time

frame_length = 256
frame_shift = 128

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=8000)
    spec = librosa.stft(y, n_fft=frame_length, hop_length=frame_shift, center=False)
    spec = np.abs(spec.T)       # N_frames * dim
    return spec, sr

def extract_label(num_frames, sr, file_path, name):
    frames_label = np.zeros(num_frames, dtype=int)
    pfile = open(file_path, 'r')
    for line in pfile.readlines():
        utt, reco, start, end = line.split()
        if(name != reco): 
            continue
        start = int(float(start)*sr/frame_shift)
        end = min(int(float(end)*sr/frame_shift), num_frames)
        frames_label[start:end] = 1
    return frames_label


if __name__ == '__main__':
    wav_dir = '../wav/'
    label_path = '../segment'
    merge_spec = []
    merge_labels = []
    for file_name in os.listdir(wav_dir):
        name, ext = os.path.splitext(file_name)
        file_path = os.path.join(wav_dir, file_name)
        spec, sr = extract_features(file_path)
        if(merge_spec == []):
            merge_spec = spec
        else:
            merge_spec = np.append(merge_spec, spec, axis=0)

        num_frames = spec.shape[0]
        labels = extract_label(num_frames, sr, label_path, name)
        if(merge_labels == []):
            merge_labels = labels
        else:
            merge_labels = np.append(merge_labels, labels)
        print(time.strftime('%Y-%m-%d %H-%M-%S', time.localtime(time.time())), 
              name + " extract spectrum and labels: success")
        print('speech percentage: {}/{}'.format(labels.sum(), labels.shape[0]))

    np.save('merge_spec.npy', merge_spec)
    np.save('merge_labels.npy', merge_labels)
    print(time.strftime('%Y-%m-%d %H-%M-%S', time.localtime(time.time())), 
          " Save all spectrum and labels: finished.")
    print('All speech percentage: {}/{}'.format(merge_labels.sum(), merge_labels.shape[0]))
