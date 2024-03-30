# -*- coding: utf-8 -*-
"""
@code :  This program implemets feature (MFCC + delta)
         extraction process for an audio. 
@Note :  20 dim MFCC(19 mfcc coeff + 1 frame log energy)
         20 dim delta computation on MFCC features. 
         20 dim delts^2 computation on MFCC features.
@output : It returns 60 dimensional feature vectors for an audio.
"""

import numpy as np
from sklearn import preprocessing
import python_speech_features as mfcc
import webrtcvad


def perform_vad(audio, rate):
    """Perform Voice Activity Detection (VAD) on the audio signal"""
    vad = webrtcvad.Vad()
    vad.set_mode(3)  # Aggressive mode

    frames = []
    frame_duration = 30  # Duration of each frame in milliseconds
    frame_size = int(rate * frame_duration / 1000)  # Number of samples in each frame

    for i in range(0, len(audio), frame_size):
        frame = audio[i:i+frame_size]
        if len(frame) < frame_size:
            frame = np.pad(frame, (0, frame_size - len(frame)), 'constant')
        if vad.is_speech(frame.tobytes(), rate):
            frames.append(frame)

    return np.concatenate(frames)


def calculate_delta(array):
    """Calculate and returns the delta of given feature vector matrix"""

    rows,cols = array.shape
    deltas = np.zeros((rows,20))
    N = 2
    for i in range(rows):
        index = []
        j = 1
        while j <= N:
            if i-j < 0:
                first = 0
            else:
                first = i-j
            if i+j > rows -1:
                second = rows -1
            else:
                second = i+j
            index.append((second,first))
            j+=1
        deltas[i] = ( array[index[0][0]]-array[index[0][1]] + (2 * (array[index[1][0]]-array[index[1][1]])) ) / 10
    return deltas

def extract_features(audio,rate, vad=False, delta=False, delta_sq=False):
    """extract 20 dim mfcc features from an audio, performs CMS and combines 
    delta to make it 40 dim feature vector"""    
    if(vad):
        audio = perform_vad(audio, rate)

    mfcc_feat = mfcc.mfcc(audio,rate, 0.025, 0.01,20,appendEnergy = True)
    
    mfcc_feat = preprocessing.scale(mfcc_feat)
    if delta:
        delta = calculate_delta(mfcc_feat)
        if delta_sq:
            delta_sq = calculate_delta(delta)
            combined = np.hstack((mfcc_feat,delta, delta_sq)) 
        else:
            combined = np.hstack((mfcc_feat,delta))
        return combined
    else:
        return mfcc_feat
    
if __name__ == "__main__":
     print("This is a module, and it should be imported to be used")
     print('In main, Call extract_features(audio,signal_rate) as parameters')
     
    