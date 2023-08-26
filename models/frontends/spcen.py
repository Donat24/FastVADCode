import torch
from torch import nn
from torch import Tensor
from typing import Optional, Tuple
from einops import rearrange
import torchaudio
from torchaudio.transforms import MelSpectrogram
from .algorithms.spcen import SPCENTrain
from .algorithms.spcen import to_script as spcen_to_script

#Train
class SPCENFrontendTrain(nn.Module):
    def __init__(self,
        
        #MEL
        frame_size = None,
        sr         = None,
        n_mels     = 40,
        mel_scale  = "slaney",

        #PCEN
        s     = 0.0001,
        alpha = 0.98,
        delta = 2,
        r     = 0.2 ,
        
    ) -> None:

        #Super
        super().__init__()

        #MEL
        self.mel = MelSpectrogram(
            sample_rate = sr,
            n_fft       = frame_size,
            n_mels      = n_mels,
            center      = False,
            mel_scale   = mel_scale,
            norm        = "slaney",
            normalized  = True,
            power       = 1.
        )

        #PCEN
        self.spcen = SPCENTrain(
            n_mels = n_mels,
            s      = s,
            alpha  = alpha,
            delta  = delta,
            r      = r
        )
    
    def forward(self, x):

        out = x
        out = self.mel(out).squeeze(-1)
        out = self.spcen(out)

        #Return
        return out

#Export
class SPCENFrontend(nn.Module):
    def __init__(self,
        
        mel   = None,
        spcen = None
        
    ) -> None:

        #Super
        super().__init__()

        self.mel   = mel
        self.spcen = spcen
    
    def forward(self, x : Tensor, m_last_frame : Optional[Tensor]) -> Tuple[Tensor, Tensor]:

        out               = x
        out               = self.mel(out).squeeze(-1)
        out, m_curr_frame = self.spcen(out,m_last_frame)

        #Return
        return out, m_curr_frame

def to_script(frontend:SPCENFrontendTrain):
    
    #Frontend
    frontend.cpu()
    frontend.eval()
    
    model = SPCENFrontend(
        mel   = frontend.mel,
        spcen = spcen_to_script(frontend.spcen),
    )

    model.cpu()
    model.eval()
    
    #Script
    model = torch.jit.script(model)

    #Return
    return model