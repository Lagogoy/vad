#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import struct


class write_feats():
    def __init__(self, file_name):
        try:
            idx_f = open(file_name + '.idx', 'w')
            self.idx_file = idx_f
            ark_f = open(file_name + '.ark', 'wb')
            self.ark_file = ark_f
            self.ark_file_name = file_name + '.ark'
        except:
            return
        
    def write(self, key, feat, label):
        pos = self.ark_file.tell()
        self.idx_file.write('%s %s:%d\n' % (key, self.ark_file_name, pos))
        
        feat = np.array(feat, dtype='float64')
        self.ark_file.write(struct.pack('I', feat.shape[0]))  # rows
        self.ark_file.write(struct.pack('I', feat.shape[1]))  # cols
        feat.tofile(self.ark_file, sep="")  # binary data
        self.ark_file.write(struct.pack('I', label))  # label
        
    def __del__(self):
        if self.idx_file:
            self.idx_file.close()
        if self.ark_file:
            self.ark_file.close()
            
            
def read_feats(feats_descriptor):
    """
    generator(key, feat, label) = read_feats(feats_descriptor)
    file_or_fd: opened file descriptor
    
    Example:
    f = open('feats_1.idx', 'r')
    for line in f:
        key, feat, label = read_feats(line)
        ....
    """
    (file_name, pos) = feats_descriptor.split(':')

    with open(file_name, 'rb') as fd:
        fd.seek(int(pos))
        rows = struct.unpack('I', fd.read(4))[0]
        cols = struct.unpack('I', fd.read(4))[0]
        buf = fd.read(rows * cols * 8)
        vec = np.frombuffer(buf, dtype='float64')
        mat = np.reshape(vec,(rows,cols))
        label = struct.unpack('I', fd.read(4))[0]
    
    return mat, label