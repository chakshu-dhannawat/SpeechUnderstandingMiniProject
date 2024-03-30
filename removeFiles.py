import os
import glob

def remove_all_files(folder):
    files = glob.glob(f'{folder}/*')
    for file in files:
        os.remove(file)

# Usage
remove_all_files(r'C:\Users\chaks\SpeechUnderstandingMiniProject\trained_models_HMM\\')