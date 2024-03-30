from removeFiles import remove_all_files

def remove_trained_models(model_types = ['GMM', 'HMM']):
    for model_type in model_types:
        remove_all_files(f'C:\\Users\\chaks\\SpeechUnderstandingMiniProject\\trained_models_{model_type}\\')