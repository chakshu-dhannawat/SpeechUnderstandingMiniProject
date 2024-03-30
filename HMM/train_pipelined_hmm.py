#This file trains the HMM models for the speakers in the development set, for various pipelines and tests them.
from train_models_hmm import train_models_hmm
from hmm_utils import remove_all_files
from hmm_utils import log

import os
from dotenv import load_dotenv
import pickle
import os
import numpy as np
from scipy.io.wavfile import read
from hmmlearn import hmm
from speakerfeatures import extract_features
import warnings
import pickle
import os
import numpy as np
from scipy.io.wavfile import read
from hmmlearn import hmm
from speakerfeatures import extract_features
import warnings
warnings.filterwarnings("ignore")

load_dotenv()

#path to training data
source   = r"C:\Users\chaks\SpeechUnderstandingMiniProject\development_set\development_set\\"
  

#path where training speakers will be saved
dest = r"C:\Users\chaks\SpeechUnderstandingMiniProject\trained_models_HMM\\"

train_file = r"C:\Users\chaks\SpeechUnderstandingMiniProject\development_set_enroll.txt"        
log("---------------------------------------------------")
# log("Training HMM models; Pipeline: MFCC+(del)MFCC+(del^2)MFCC")
train_models_hmm(source, dest, train_file, vad=False, delta=False, delta_sq=False)