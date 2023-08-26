from .datasets import *

from IPython.display import Audio
from IPython.display import clear_output, display

def get_audio(dataset, idxs):
    
    #lädt UnchunkedDataset
    if isinstance(dataset, ChunkedDataset):
        dataset = dataset.dataset

    #Cast
    if isinstance(idxs, int):
        idxs = [idxs]
    #Iter
    for idx in idxs:
        x,y = dataset[idx]

        #Iterriert bis Samplerate gefunden wird
        sr_dataset = dataset
        while not hasattr(sr_dataset,"target_samplerate"):
            sr_dataset = sr_dataset.dataset

        #Audio
        display(Audio(data=x, rate=sr_dataset.target_samplerate))