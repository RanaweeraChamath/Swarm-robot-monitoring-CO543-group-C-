# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 20:14:27 2020

@author: Kushani Ranaweera
"""
import os
import numpy as np
from keras.optimizers import Adam
from EDVR_arch import EDVR, charbonnier_penalty
import cv2
import matplotlib.pyplot as plt
#matplotlib inline

path = 'C:/Users/Kushani Ranaweera/Documents/tharka/6th semester/CO543/project/Implenentations/case3_2.mp4'
images = os.listdir(path+'GT')
nframes = 5
center = nframes//2
batch_size = 1

X = []
y = []
for k, img in enumerate(images):
    y.append(cv2.imread(os.path.join(path, 'GT', img)))
    X.append(cv2.imread(os.path.join(path, 'blur_bicubic', img)))
X = np.stack(X)
y = np.stack(y)

X_trn = []
for i in range(len(images)):
    next_frames = X[i:i+center+1]
    if i<center:
        prev_frames = X[:i]
    else:
        prev_frames = X[i-center:i]
    
    to_fill = nframes - next_frames.shape[0] - prev_frames.shape[0]
    if to_fill:
        if len(prev_frames) and i<nframes:
            pad_x = np.repeat(prev_frames[0][None], to_fill, axis=0)
            xx = np.concatenate((pad_x, prev_frames, next_frames))
        else:
            if i>nframes:
                pad_x = np.repeat(next_frames[-1][None], to_fill, axis=0)
                xx = np.concatenate((prev_frames, next_frames, pad_x))
            else:
                pad_x = np.repeat(next_frames[0][None], to_fill, axis=0)
                xx = np.concatenate((pad_x, prev_frames, next_frames))
    else:
        xx = np.concatenate((prev_frames, next_frames))
    X_trn.append(xx)
X_trn = np.stack(X_trn)
X_trn.shape, y.shape

HR_in = False if np.prod(X_trn.shape[2:]) < np.prod(y.shape[1:]) else True
print(HR_in)

VideoSuperResolution = EDVR(inp_shape=X_trn.shape[2:],
                            nf=64, nframes=nframes,
                            groups=8, front_RBs=5,
                            back_RBs=10, center=None,
                            predeblur=True, HR_in=HR_in)
model = VideoSuperResolution.get_EDVR_model()
optimizer = Adam(lr=4e-4, beta_1=.9, beta_2=0.999)

model.compile(optimizer=optimizer, loss=charbonnier_penalty)
model.fit(x=X_trn/255, y=y/255, batch_size=batch_size,
          epochs=5, validation_split=0.1, shuffle=True)

