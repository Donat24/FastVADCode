import torch
from torch import nn
import torchaudio

from .util import *

#Gain Util
class Gain(nn.Module):
    def __init__(self ) -> None:
        super().__init__()
    
    @torch.no_grad()
    def forward(self, x, gain = 0):
        
        x = x * db_to_amp(gain)
        
        #HardClipper
        x[x > 1]  = 1
        x[x < -1] = -1
        
        return x


#Modul welches Waveform random gained
class RandomGain(Gain):
    def __init__(self, low = -20, high = 6) -> None:
        super().__init__()
        
        #Params
        self.low = low
        self.high = high
    
    @torch.no_grad()
    def forward(self, x):
        return super().forward(x, gain = torch.randint(self.low, self.high, size=(1,)))