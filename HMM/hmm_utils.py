import os
import glob

def remove_all_files(folder =r'C:\Users\chaks\SpeechUnderstandingMiniProject\trained_models_HMM\\' ):
    files = glob.glob(f'{folder}/*')
    for file in files:
        os.remove(file)

# remove_all_files()
def log(string, file_path= r"C:\Users\chaks\SpeechUnderstandingMiniProject\HMM\hmm_log.txt"):
    with open(file_path, 'a') as file:
        file.write(string + '\n')