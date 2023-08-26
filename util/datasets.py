import librosa
import torch
import os
import pandas as pd
from torch.utils.data import Dataset
import tarfile
import typing
import math
import numpy as np

from .util import *

class BaseDataset(Dataset):
    def __init__(self, csv_file = None, data = None, target_samplerate=16000, fixed_length = False) -> None:
        super().__init__()

        #Speichert Data direkt
        if data is not None:
            
            if "filename" not in data.columns:
                raise Exception ("Bad Data Format")
            
            self.data = data

        #Lädt Csv
        elif csv_file is not None:
            
            data = pd.read_csv(csv_file)

            if "filename" not in data.columns:
                raise Exception ("Bad Data Format")
            
            self.data = data
        
        #Für Transforms
        self.target_samplerate = target_samplerate
        self.fixed_length = fixed_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, n):

        #lädt Zeile
        row = self.data.iloc[n]

        #lädt Datei
        waveform, sample_rate = librosa.load(self.__getfile__(row.filename), sr=self.target_samplerate, mono=True, dtype="float64")

        #Tensor
        waveform = torch.from_numpy(waveform).to(torch.float32)
        
        #fixed Length
        if self.fixed_length:

            #Waveform zu Lang
            if waveform.size(-1) > self.fixed_length:
                waveform = waveform[ : self.fixed_length]

        #Return
        return waveform, self.target_samplerate, row
        
    #Abstract
    def __getfile__(self, filename):
        raise Exception("__getfile__ not implemented")
    
    def check_files(self, rms_treshhold = 0.001):
        try:
            for waveform, _, row in self:
                if rms(waveform) < rms_treshhold:
                    raise Exception(f"{row.filename} has rms < {rms_treshhold}")
        
        except Exception as e:
            print(e)

#Lädt Audio-Datein aus Ordner und dazugeörigen CSV-Eintrag
class LocalFileDataset(BaseDataset):
    def __init__(self, root_dir, csv_file=None, data=None, target_samplerate=16000, fixed_length = False) -> None:
        
        #Verzeichnissordner
        self.root_dir    = root_dir

        #Erstellt DataFrame aus Datein im Ordner
        if csv_file is None and data is None:
            files = [entry.name for entry in os.scandir(self.root_dir)]
            df    = pd.DataFrame({"filename" : files})
            df    = df[df.filename.str.contains("wav|mp3|ogg|flac")]
            data  = df
        
        #Super
        super().__init__(csv_file, data, target_samplerate, fixed_length)

    def __getfile__(self, filename):
        return os.path.join(self.root_dir, filename)

#Lädt Audio-Datein aus TAR und dazugeörigen CSV-Eintrag
class TarDataset(BaseDataset):
    def __init__(self, tar_file, csv_file=None, data=None, target_samplerate=16000, fixed_length = False) -> None:
        
        #Tar
        self.tar_file         = tar_file
        self.tar_file_handler = None

        #Super
        super().__init__(csv_file, data, target_samplerate, fixed_length)

    #Öffnet Datei
    def open(self):
        if self.tar_file_handler is None:
            self.tar_file_handler = tarfile.open(self.tar_file, mode="r")
    
    def close(self):
        if self.tar_file_handler is not None:
            self.tar_file_handler.close()
    
    def __getfile__(self, filename):
        #opens file
        if self.tar_file_handler is None:
            self.open()

        #liest teil
        member = self.tar_file_handler.getmember(filename)
        return self.tar_file_handler.extractfile(member)
    
    def __del__(self):
        self.close()

#Returned Waveform, Y
class SpeakDataset(Dataset):
    def __init__(self, dataset, audio_processing_chain, get_y, normalizer = None) -> None:
        super().__init__()

        #Parameter
        self.dataset                = dataset
        self.audio_processing_chain = audio_processing_chain
        self.get_y                  = get_y
        self.normalizer             = normalizer
    
    def __len__(self):
            return len(self.dataset)

    @torch.no_grad()
    def __getitem__(self, idx):
        
        #Lädt x
        x, sr, info = self.dataset[idx]

        #Erzeugt y
        y = self.get_y(x,sr,info)

        #Audio Processing
        if self.audio_processing_chain is not None:
            x = self.audio_processing_chain(x, sr, info)
        
        #Normalizer
        if self.normalizer is not None:
            x = self.normalizer(x)
        
        return x, y

class SplitDataset(Dataset):
    def __init__(self, dataset, max_length, part_lenth) -> None:
        super().__init__()

        #Dataset
        self.dataset     = dataset
        
        #Params
        self.max_length       = max_length
        self.part_lenth       = part_lenth
        self.parts_per_sample = self.max_length // self.part_lenth
    
    def __len__(self):
        return len(self.dataset) * self.parts_per_sample
    
    @torch.no_grad()
    def __getitem__(self, idx):
        
        #IDX
        file_idx = idx // self.parts_per_sample
        part_idx = idx %  self.parts_per_sample

        #lädt Datei
        x, y = self.dataset[file_idx]

        #Start und Ende
        start_idx = part_idx  * self.part_lenth
        end_idx   = start_idx + self.part_lenth

        #Return
        return x[start_idx:end_idx], y[start_idx:end_idx]
    
    @torch.no_grad()
    def __getitem__(self, idx):
        return self.dataset[self.subset_idx[idx]]

class ReshapeFileDataset(Dataset):
    def __init__(self, dataset, noise_edge = 0.5, sr = None, ratio = 1) -> None:
        super().__init__()
        self.dataset    = dataset
        self.noise_edge = librosa.time_to_samples( times = noise_edge, sr = sr)
        self.ratio      = ratio
    
    def __len__(self):
        return len(self.dataset)

    @torch.no_grad()
    def __getitem__(self, idx):
        
        # lädt Dataset
        x, y = self.dataset[idx]

        # Switch
        if self.ratio is None or self.ratio <= 0:
            return x, y
        
        _non_zero = y.nonzero()
        if _non_zero.sum() < 2 :
            return x, y
        
        start     = _non_zero[ 0].item()
        end       = _non_zero[-1].item()
        voice_len = end - start
        new_size  = voice_len * (self.ratio + 1)

        #neuer Start und Ende
        max_start = max(start - self.noise_edge, 0)
        min_end   = min(end   + self.noise_edge, x.size(-1))
        min_start = max(min_end - new_size, 0)
        min_start = min(min_start, max_start)
        if max_start > 0 and max_start > min_start:
            new_start = torch.randint(low = min_start, high = max_start, size=(1,))
        else:
            new_start = max_start
        new_end   = min(new_start + new_size, x.size(-1))

        #Return
        return x[new_start:new_end], y[new_start:new_end]


#Chunker
class ChunkedDataset(Dataset):
    def __init__(self, dataset, sample_length, hop_length, context_length = 0, fill_x_to_sample_length = True, fill_y_to_sample_length = True, chunk_y = True, y_truth_treshold:typing.Union[float, int] = 0, sample_processing_chain = None, max_frames = None) -> None:
        super().__init__()

        #unchunked Dataset
        self.dataset = dataset

        #Chunker Parameter
        self.sample_length  = sample_length
        self.hop_length     = hop_length
        self.context_length = context_length

        #Dateigröße auf vollen Chunk anpassen
        self.fill_x_to_sample_length = fill_x_to_sample_length
        self.fill_y_to_sample_length = fill_y_to_sample_length

        #Chunk Y
        self.chunk_y = chunk_y

        #sample_processing_chain
        self.sample_processing_chain = sample_processing_chain

        #Truth Treshold
        if isinstance(y_truth_treshold, float):
            self.y_truth_treshold = math.floor(y_truth_treshold * self.sample_length)
        
        elif isinstance(y_truth_treshold, int):
            self.y_truth_treshold = y_truth_treshold
        
        #MAX FRAMES
        self.max_frames = max_frames
    
    def __len__(self):
            return len(self.dataset)

    @torch.no_grad()
    def __getitem__(self, idx):
        
        #Lädt x, y für 
        x, y = self.dataset[idx]

        #Füllt Sample Länge auf
        if self.fill_x_to_sample_length or self.fill_y_to_sample_length:
            
            #Errechnet
            _overflow = (x.size(-1) - self.sample_length) % self.hop_length
            
            if _overflow:
                
                #Erzeugt 0en zum Auffüllen
                _length_zeros = self.hop_length - _overflow
                zeros = torch.zeros(size = (_length_zeros,))

                #Füllt X und Y auf
                if self.fill_x_to_sample_length:
                    x = torch.concat([x, zeros])

                if self.fill_y_to_sample_length:
                    y = torch.concat([y, zeros])
        
        #Fügt 0en am Anfang hinzu für Context
        if self.context_length:
            zeros = torch.zeros(size = (self.context_length,))
            x = torch.concat([zeros, x])
        
        #Erzeugt X
        x = get_samples(x, self.context_length + self.sample_length, self.hop_length)

        #Erzeugt y
        if self.chunk_y:
            y = get_samples(y, self.sample_length, self.hop_length)
            y = y.sum(dim=-1).gt_(self.y_truth_treshold)
        
        #sample_processing_chain
        if self.sample_processing_chain:
            x = self.sample_processing_chain(x)
        
        #Kürzt Frames
        if self.max_frames:
            if x.size(0) > self.max_frames:
                diff = x.size(0) - self.max_frames
                rand_start = torch.randint(low=0, high=diff, size=(1,))
                x = x[rand_start : rand_start + self.max_frames]
                y = y[rand_start : rand_start + self.max_frames]

        return x, y