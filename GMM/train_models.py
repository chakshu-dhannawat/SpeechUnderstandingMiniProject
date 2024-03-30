#train_models.py

import pickle
import os
import numpy as np
from scipy.io.wavfile import read
from sklearn import mixture
from speakerfeatures import extract_features
import warnings
warnings.filterwarnings("ignore")
from gmm_utils import log

source   = r"C:\Users\chaks\SpeechUnderstandingMiniProject\development_set\development_set\\"
  
#path where training speakers will be saved
dest = r"C:\Users\chaks\SpeechUnderstandingMiniProject\trained_models_GMM\\"

train_file = r"C:\Users\chaks\SpeechUnderstandingMiniProject\development_set_enroll.txt"        


file_paths = open(train_file,'r')

count = 1


# log("Training GMM | Pipeline: MFCC+ MFCC\' + MFCC\'\'")
# Extracting features for each speaker (5 files per speakers)
features = np.asarray(())
for path in file_paths:    
    path = path.strip()   
    print(path)
    
    # read the audio
    sr,audio = read(source + path)
    
    # extract 40 dimensional MFCC & delta MFCC features
    vector   = extract_features(audio,sr)
    
    if features.size == 0:
        features = vector
    else:
        features = np.vstack((features, vector))
    # when features of 5 files of speaker are concatenated, then do model training
    if count == 5:    
        gmm = mixture.GaussianMixture(n_components = 16, max_iter = 200, covariance_type='diag',n_init = 3)
        gmm.fit(features)
        picklefile = path.split("-")[0] + ".gmm"
        picklepath = os.path.join(dest, picklefile)  # Corrected path for each speaker model
        if not os.path.isdir(os.path.dirname(picklepath)):
            os.makedirs(os.path.dirname(picklepath))

        with open(picklepath, 'wb') as file:
            pickle.dump(gmm, file)
        print('+ modeling completed for speaker:', picklefile, " with data point = ", features.shape)
        # dumping the trained gaussian model
        # picklefile = path.split("-")[0]+".gmm"
        # cPickle.dump(gmm,open(dest + picklefile,'w'))
        # print '+ modeling completed for speaker:',picklefile," with data point = ",features.shape    
        features = np.asarray(())
        count = 0
    count = count + 1
    