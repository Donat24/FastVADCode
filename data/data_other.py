import ast
import pandas as pd

from .data_config import *
from .paths_config import *
from util.datasets import *
from util.audio_processing import *
from .audio_processing_chains import AudioProcessing

# AVA #

#Liest CSV
data_ava_train          = pd.read_csv(AVA_TRAIN_CSV_PATH)
data_ava_train_splitted = pd.read_csv(AVA_TRAIN_CSV_SPLITTED_PATH)
data_ava_test           = pd.read_csv(AVA_TEST_CSV_PATH)
data_ava_train["parts"]          = data_ava_train.parts.apply(ast.literal_eval)
data_ava_train_splitted["parts"] = data_ava_train_splitted.parts.apply(ast.literal_eval)
data_ava_test["parts"]           = data_ava_test.parts.apply(ast.literal_eval)

#Erstellt Y-Tensor
def get_y_ava(tensor, sr ,info):
    out = torch.zeros_like(tensor)
    
    #Iterriert Parts
    for label, start, end in info.parts:
        
        if not label == "NO_SPEECH":
            start = librosa.time_to_samples(times=start, sr=sr)
            end   = librosa.time_to_samples(times=end,sr=sr)
            out[start:end] = 1
    
    return out

#Datasets
filedataset_ava_train           = LocalFileDataset(root_dir=AVA_DIR_PATH, data=data_ava_train)
filedataset_ava_train_splitted  = LocalFileDataset(root_dir=AVA_SPLITTED_DIR_PATH, data=data_ava_train_splitted)
filedataset_ava_test            = LocalFileDataset(root_dir=AVA_DIR_PATH, data=data_ava_test)
speakdataset_ava_train          = SpeakDataset(filedataset_ava_train,          audio_processing_chain=None, get_y = get_y_ava)
speakdataset_ava_train_splitted = SpeakDataset(filedataset_ava_train_splitted, audio_processing_chain=None, get_y = get_y_ava)
speakdataset_ava_test           = SpeakDataset(filedataset_ava_test,           audio_processing_chain=None, get_y = get_y_ava)
dataset_ava_train_splitted      = ChunkedDataset(speakdataset_ava_train_splitted, SAMPLE_LENGTH, HOP_LENGTH, CONTEXT_LENGTH, y_truth_treshold = TRUTH_TRESHOLD, max_frames = MAX_FRAMES)
dataset_ava_test                = ChunkedDataset(speakdataset_ava_test,  SAMPLE_LENGTH, HOP_LENGTH, CONTEXT_LENGTH, chunk_y = False, fill_y_to_sample_length = False)

#LibriParty
data_LibriParty_val           = pd.read_csv(LIBRIPARTY_VAL_CSV_PATH)
data_LibriParty_val_splitted  = pd.read_csv(LIBRIPARTY_VAL_SPLITTED_CSV_PATH)
data_LibriParty_test          = pd.read_csv(LIBRIPARTY_TEST_CSV_PATH)
data_LibriParty_val["parts"]           = data_LibriParty_val.parts.apply(ast.literal_eval)
data_LibriParty_val_splitted["parts"]  = data_LibriParty_val_splitted.parts.apply(ast.literal_eval)
data_LibriParty_test["parts"]          = data_LibriParty_test.parts.apply(ast.literal_eval)

#Erstellt Y-Tensor
def get_LibriParty_chime(tensor, sr ,info):
    out = torch.zeros_like(tensor)
    
    #Iterriert Parts
    for start, end in info.parts:
        start = librosa.time_to_samples(times=start, sr=sr)
        end   = librosa.time_to_samples(times=end,sr=sr)
        out[start:end] = 1
    
    return out

filedataset_LibriParty_val            = LocalFileDataset(root_dir=LIBRIPARTY_VAL_DIR,  data=data_LibriParty_val)
filedataset_LibriParty_val_splitted   = LocalFileDataset(root_dir=LIBRIPARTY_VAL_SPLITTED_DIR,  data=data_LibriParty_val_splitted)
filedataset_LibriParty_test           = LocalFileDataset(root_dir=LIBRIPARTY_TEST_DIR, data=data_LibriParty_test)
speakdataset_LibriParty_val           = SpeakDataset(filedataset_LibriParty_val,  audio_processing_chain=None,  get_y = get_LibriParty_chime)
speakdataset_LibriParty_val_splitted  = SpeakDataset(filedataset_LibriParty_val_splitted,  audio_processing_chain=None,  get_y = get_LibriParty_chime)
speakdataset_LibriParty_test = SpeakDataset(filedataset_LibriParty_test, audio_processing_chain=None,  get_y = get_LibriParty_chime)
dataset_LibriParty_val       = ChunkedDataset(speakdataset_LibriParty_val_splitted, SAMPLE_LENGTH, HOP_LENGTH, CONTEXT_LENGTH, y_truth_treshold = TRUTH_TRESHOLD)
dataset_LibriParty_test      = ChunkedDataset(speakdataset_LibriParty_test,        SAMPLE_LENGTH, HOP_LENGTH, CONTEXT_LENGTH, chunk_y = False, fill_y_to_sample_length = False)