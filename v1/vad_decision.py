import os
import numpy as np
import librosa
from scipy import signal
import convnet

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader


root = os.getcwd()
wav_dir = '/home/linqingjian/src/bin/diarization/wav'
vad_dir = '/home/linqingjian/src/bin/diarization/vad'
# wav_dir = os.path.join(root, 'wav')
# vad_dir = os.path.join(root, 'vad')
model_path = os.path.join(root, 'model_best.pth.tar')
label_path = os.path.join(root, 'seg', 'segment')

def main():
    model = convnet.TinyConvNet(2)
    print(model)
    model = torch.nn.DataParallel(model).cuda()
    
    # Model loading
    if os.path.isfile(model_path):
        print("==> Loading best model...")
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['state_dict'])
        print("==> Load best model: success!")
    else:
        print("==> no model found, EXIT")
        return
    
    wav_files = ['iaaa.wav', 'iaab.wav', 'iaac.wav', 'iaad.wav', 'iaae.wav', 
                 'iaaf.wav', 'iaag.wav', 'iaah.wav', 'iaai.wav', 'iaaj.wav']
    
    frame_len = 0.025
    frame_shift = 0.01
    ext_frames = 15

    model.eval()   
    for wavfile in wav_files:
        name, ext = os.path.splitext(wavfile)
        vad_flags = open(os.path.join(vad_dir, name+'.vad'),'w')
        
        eval_dataset = HADDataset(wavfile, label_path, frame_len, frame_shift, ext_frames)
        eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=256, 
                          shuffle=False, num_workers=4, pin_memory=True)
        correct = 0
        false_alarm = 0
        miss_detect = 0
        for i, (feats, label) in enumerate(eval_loader):
            label = label.cuda(async=True)
            input_var = torch.autograd.Variable(feats, volatile=True)
            # compute output
            output = model(input_var)
            _, pred = output.data.topk(k=1, dim=1, largest=True)
            pred = torch.squeeze(pred)
            for j in range(pred.size()[0]):
                vad_flags.write(str(pred[j]))
            vector = pred - label
            false_alarm += (vector==1).float().sum()
            miss_detect += (vector==-1).float().sum()
            correct += (vector==0).float().sum()
        vad_flags.close()
        
        total = correct + false_alarm + miss_detect
        correct /= total
        false_alarm /= total
        miss_detect /= total
        print('File {}:\t correct:{}\t false_alarm:{}\t miss_detect:{}\t'.format(
            wavfile, correct, false_alarm, miss_detect))
        

class HADDataset(Dataset):
    def __init__(self, wavfile, label_path, frame_len, frame_shift, ext_frames):
        wav_path = os.path.join(wav_dir, wavfile)
        spec, sr = log_mel_fbanks_energy(wav_path, frame_len, frame_shift)
        # spec shape: channel * feature_dim * num_frames
        name, ext = os.path.splitext(wavfile)
        num_frames = spec.shape[2]
        frame_label = extract_label(num_frames, frame_shift, label_path, name)
        
        self.ext_frames = ext_frames
        self.spec = spec
        self.frame_label = frame_label
        print('{} frames in {}'.format(spec.shape[2], wavfile))

    def __getitem__(self, index):
        num_frames = self.spec.shape[2]
        frame_idxs = [0 if i < 0 else i if i < num_frames else num_frames-1 
                     for i in range(index-self.ext_frames, index+self.ext_frames+1)]
        return self.spec[:, :, frame_idxs], self.frame_label[index]

    def __len__(self):
        return self.spec.shape[2]
    

def log_mel_fbanks_energy(file_path, frame_len, frame_shift):
    y, sr = librosa.load(file_path, sr=8000)
    y = signal.lfilter([1, -0.97], 1, y)
    spec = librosa.stft(y, n_fft=int(frame_len*sr), hop_length=int(frame_shift*sr), center=False)
    spec = librosa.feature.melspectrogram(S=np.abs(spec)**2+1e-10, n_mels=32)
    spec = cmvn(np.log(spec + 10e-10))
    spec_delta = cmvn(librosa.feature.delta(spec))
    spec_delta2 = cmvn(librosa.feature.delta(spec, order=2))
    spec = np.concatenate([np.array([spec]), np.array([spec_delta]), 
                           np.array([spec_delta2])], axis=0)
    spec = spec.astype(np.float32)
    return spec, sr

def cmvn(spectrogram):
    '''Cepstral Mean and Variance Normalization
    '''
    mu = np.mean(spectrogram, axis=1)
    stdev = np.std(spectrogram, axis=1)
    return (spectrogram - mu.reshape((-1, 1))) / stdev.reshape((-1, 1))

def extract_label(num_frames, frame_shift, file_path, name):
    frames_label = np.zeros(num_frames, dtype=int)
    pfile = open(file_path, 'r')
    for line in pfile.readlines():
        utt, reco, start, end = line.split()
        if(name != reco):
            continue
        start = int(float(start)/frame_shift)
        end = min(int(float(end)/frame_shift), num_frames)
        frames_label[start:end] = 1
    return frames_label


if __name__ == '__main__':
    main()