#test_gender.py
import os
import pickle
import numpy as np
from scipy.io.wavfile import read
from speakerfeatures import extract_features
import warnings
warnings.filterwarnings("ignore")
import time
from hmm_utils import log
from hmm_utils import remove_all_files

#path to training data
source   = r"C:\Users\chaks\SpeechUnderstandingMiniProject\development_set\development_set\\"   

modelpath = r"C:\Users\chaks\SpeechUnderstandingMiniProject\trained_models_HMM\\"

test_file = r"C:\Users\chaks\SpeechUnderstandingMiniProject\development_set_test.txt"        

file_paths = open(test_file, 'r', encoding='utf-8')


hmm_files = [os.path.join(modelpath,fname) for fname in 
              os.listdir(modelpath) if fname.endswith('.hmm')]

#Load the Gaussian gender Models
models = [pickle.load(open(fname, 'rb')) for fname in hmm_files]
speakers   = [fname.split("\\")[-1].split(".hmm")[0] for fname 
              in hmm_files]

# Read the test directory and get the list of test audio files 
# for path in file_paths:   
    
#     path = path.strip()   
#     print(path)
#     sr,audio = read(source + path)
#     vector   = extract_features(audio,sr)
    
#     log_likelihood = np.zeros(len(models)) 
    
#     for i in range(len(models)):
#         gmm    = models[i]         #checking with each model one by one
#         scores = np.array(gmm.score(vector))
#         log_likelihood[i] = scores.sum()
    
#     winner = np.argmax(log_likelihood)
#     print("\tdetected as - ", speakers[winner])
#     time.sleep(1.0) 


total_files = 0
correct_predictions = 0

for path in file_paths:
    path = path.strip()
    print(path)
    sr, audio = read(os.path.join(source, path))  # Fix path concatenation
    vector = extract_features(audio, sr)

    log_likelihood = np.zeros(len(models))

    for i in range(len(models)):
        hmm = models[i]  # checking with each model one by one
        scores = np.array(hmm.score(vector))
        log_likelihood[i] = scores.sum()

    winner = np.argmax(log_likelihood)
    print("\tdetected as - ", speakers[winner])

    # Extract speaker name from the path
    speaker_from_path = path.split("-")[0]
    # Extract detected speaker name from the model
    detected_speaker = speakers[winner]

    # Check accuracy
    if speaker_from_path == detected_speaker:
        correct_predictions += 1

    total_files += 1

    time.sleep(1.0)

accuracy = correct_predictions / total_files * 100
log("Accuracy on test data")
log("Total accuracy: {:.2f}%".format(accuracy))
log("")
print("Total accuracy: {:.2f}%".format(accuracy))
# remove_all_files()