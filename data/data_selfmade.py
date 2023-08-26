import ast
import librosa
import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
    
from .data_config import *
from .paths_config import *
from .audio_processing_chains import *
from util.datasets import *
from util.audio_processing import *

#Lädt CSVs
train_csv = pd.read_csv(TRAIN_CSV_PATH)
test_csv  = pd.read_csv(TEST_CSV_PATH)
val_csv   = pd.read_csv(VAL_CSV_PATH)

#FileDataset
filedataset_train = TarDataset(TRAIN_TAR_PATH, data=train_csv, target_samplerate=SAMPLE_RATE)
filedataset_test  = TarDataset(TEST_TAR_PATH,  data=test_csv,  target_samplerate=SAMPLE_RATE)
filedataset_val   = TarDataset(TRAIN_TAR_PATH, data=val_csv,   target_samplerate=SAMPLE_RATE)

#AudioProcessingChain
audio_processing_chain_tain = AudioProcessingTrain()
audio_processing_chain_val  = AudioProcessingTest()
audio_processing_chain_test = AudioProcessingTest()

#Erstellt Y-Tensor
def get_y(tensor, sr ,info):
    out = torch.zeros_like(tensor)
    out[info["start"] : info["end"] + 1] = 1
    return out

#SpeakDataset
speakdataset_train_unchunked = SpeakDataset(filedataset_train, audio_processing_chain = audio_processing_chain_tain, get_y = get_y)
speakdataset_test_unchunked  = SpeakDataset(filedataset_test,  audio_processing_chain = audio_processing_chain_test, get_y = get_y)
speakdataset_val_unchunked   = SpeakDataset(filedataset_val,   audio_processing_chain = audio_processing_chain_val,  get_y = get_y)

#RESHAPED
speakdataset_train_reshaped  = ReshapeFileDataset(speakdataset_train_unchunked, sr = SAMPLE_RATE, ratio = SPEECH_RATIO)

#ChunkedDataset
dataset_train = ChunkedDataset(speakdataset_train_reshaped,  SAMPLE_LENGTH, HOP_LENGTH, CONTEXT_LENGTH, y_truth_treshold = TRUTH_TRESHOLD, max_frames = MAX_FRAMES)
dataset_val   = ChunkedDataset(speakdataset_val_unchunked,   SAMPLE_LENGTH, HOP_LENGTH, CONTEXT_LENGTH, y_truth_treshold = TRUTH_TRESHOLD)
dataset_test  = ChunkedDataset(speakdataset_test_unchunked,  SAMPLE_LENGTH, HOP_LENGTH, CONTEXT_LENGTH, chunk_y = False, fill_y_to_sample_length = False)

#SPEECHCOMMANDS
speechcommands_csv               = pd.read_csv(SPEECHCOMMANDS_CSV_PATH)
filedataset_speechcommands_train = LocalFileDataset(SPEECHCOMMANDS_DIR_PATH, data=speechcommands_csv, target_samplerate=SAMPLE_RATE)
speakdataset_speechcommands      = SpeakDataset(filedataset_speechcommands_train, None, get_y)
dataset_speechcommands           = ChunkedDataset(speakdataset_speechcommands,  SAMPLE_LENGTH, HOP_LENGTH, CONTEXT_LENGTH, y_truth_treshold = TRUTH_TRESHOLD, max_frames = MAX_FRAMES)

#Common Voice
def get_y_common_voice(tensor, sr ,info):
    out = torch.zeros_like(tensor)
    for part in info["parts"]:
        start = math.floor(part[0] / 1000 * sr)
        end   = math.floor(part[1] / 1000 * sr)
        out[start : end] = 1
    return out

common_voice_german_csv           = pd.read_csv(COMMON_VOICE_GERMAN_CSV_PATH)
common_voice_german_csv["parts"]  = common_voice_german_csv["parts"].apply(ast.literal_eval)
filedataset_common_voice_german   = LocalFileDataset(COMMON_VOICE_GERMAN_DIR_PATH, data=common_voice_german_csv, target_samplerate=SAMPLE_RATE)
speakdataset_common_voice_german  = SpeakDataset(filedataset_common_voice_german, None, get_y_common_voice)
dataset_common_voice_german       = ChunkedDataset(speakdataset_common_voice_german,  SAMPLE_LENGTH, HOP_LENGTH, CONTEXT_LENGTH, y_truth_treshold = TRUTH_TRESHOLD, max_frames = MAX_FRAMES)

common_voice_english_csv           = pd.read_csv(COMMON_VOICE_ENGLISH_CSV_PATH)
common_voice_english_csv["parts"]  = common_voice_english_csv["parts"].apply(ast.literal_eval)
filedataset_common_voice_english   = LocalFileDataset(COMMON_VOICE_ENGLISH_DIR_PATH, data=common_voice_english_csv, target_samplerate=SAMPLE_RATE)
speakdataset_common_voice_english  = SpeakDataset(filedataset_common_voice_english, None, get_y_common_voice)
dataset_common_voice_english       = ChunkedDataset(speakdataset_common_voice_english,  SAMPLE_LENGTH, HOP_LENGTH, CONTEXT_LENGTH, y_truth_treshold = TRUTH_TRESHOLD, max_frames = MAX_FRAMES)
