#!/usr/bin/env python

from python_speech_features import mfcc
from python_speech_features import logfbank
from python_speech_features import delta
import scipy.io.wavfile as wav
from sklearn import preprocessing
import numpy as np

(rate,sig) = wav.read("rihanna.wav")
mfcc_feat = mfcc(sig, rate, 1./10, 1./15)
mfcc_delta = delta(mfcc_feat, 2)
mfcc_delta2 = delta(mfcc_delta,2)

mfcc_delta = np.array(mfcc_delta)
mfcc_delta2 = np.array(mfcc_delta2)


min_max_scaler = preprocessing.MinMaxScaler()


mfcc_feat = min_max_scaler.fit_transform(mfcc_feat)
mfcc_delta = min_max_scaler.fit_transform(mfcc_delta)
mfcc_delta2 = min_max_scaler.fit_transform(mfcc_delta2)


print mfcc_feat.shape
print mfcc_delta.shape
print mfcc_delta2.shape

mfcc = np.concatenate((mfcc_feat, mfcc_delta, mfcc_delta2), axis=1)

print mfcc.shape


mfcc_7200 = mfcc[:7200]
mfcc_only_7200 = mfcc_feat[:7200]

np.save("mfcc_only_7200", mfcc_only_7200)
np.save("mfcc_7200", mfcc_7200)

