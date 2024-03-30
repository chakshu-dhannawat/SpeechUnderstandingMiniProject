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

load_dotenv()  # take environment variables from .env.

# source = os.getenv('TRAIN_DB')
# dest = os.getenv('MODELS_HMM')
# train_file = os.getenv('TRAIN_PATHS_TXT')


def train_models_hmm(source, dest, train_file, vad, delta, delta_sq):
    warnings.filterwarnings("ignore")

    file_paths = open(train_file,'r')

    count = 1

    # Extracting features for each speaker (5 files per speakers)
    features = np.asarray(())
    for path in file_paths:    
        path = path.strip()   
        print(path)

        # read the audio
        sr,audio = read(source + path)

        # extract features
        vector = extract_features(audio, sr, vad, delta, delta_sq)

        if features.size == 0:
            features = vector
        else:
            features = np.vstack((features, vector))
        # when features of 5 files of speaker are concatenated, then do model training
        if count == 5:    
            gmm = hmm.GaussianHMM(n_components=16, covariance_type='full')
            gmm.fit(features)
            picklefile = path.split("-")[0] + ".hmm"
            picklepath = os.path.join(dest, picklefile)  # Corrected path for each speaker model
            if not os.path.isdir(os.path.dirname(picklepath)):
                os.makedirs(os.path.dirname(picklepath))

            with open(picklepath, 'wb') as file:
                pickle.dump(gmm, file)
            print('+ modeling completed for speaker:', picklefile, " with data point =", features.shape)
            features = np.asarray(())
            count = 0
        count = count + 1