import torch
from torch import nn
from torch import Tensor
from typing import Optional, Tuple
from einops import rearrange
import torchaudio
from .spcen import SPCENFrontendTrain
from .spcen import to_script as spcen_to_script

#Train
class SPCENCCFrontendTrain(nn.Module):
    def __init__(self,
        
        #MEL
        frame_size = None,
        sr         = None,
        n_mels     = 64,
        mel_scale  = "slaney",

        #PCEN
        s     = 0.0001,
        alpha = 0.98,
        delta = 2,
        r     = 0.2 ,

        #CCs
        n_cc       = 20,
        drop_first = True, 
        
    ) -> None:

        #Super
        super().__init__()

        #MEL
        self.spcen = SPCENFrontendTrain(
            frame_size  = frame_size,
            sr          = sr,
            n_mels      = n_mels,
            mel_scale   = mel_scale,
            s           = s,
            alpha       = alpha,
            delta       = delta,
            r           = r
        )

        #DCT
        self.register_buffer( "dct", torchaudio.functional.create_dct(
            n_mfcc = n_cc,
            n_mels = n_mels,
            norm   = "ortho"
        ))

        #Drop first
        self.drop_first = drop_first

        
    
    def forward(self, x):

        out = x
        out = self.spcen(out)
        out = torch.matmul(out, self.dct)

        #Drops first
        if self.drop_first:
            out = out[...,1:]

        #Return
        return out

#Export
class SPCENCCFrontend(nn.Module):
    def __init__(self,
        
        spcen      = None,
        dct        = None,
        drop_first = None,
        
    ) -> None:

        #Super
        super().__init__()

        self.spcen = spcen
        self.register_buffer("dct", dct)
        self.drop_first = drop_first
    
    def forward(self, x : Tensor, m_last_frame : Optional[Tensor]) -> Tuple[Tensor, Tensor]:

        out               = x
        out, m_last_frame = self.spcen(out, m_last_frame)
        out               = torch.matmul(out, self.dct)

        #Drops first
        if self.drop_first:
            out = out[...,1:]

        #Return
        return out, m_last_frame

def to_script(frontend:SPCENFrontendTrain):
    
    #Frontend
    frontend.cpu()
    frontend.eval()
    
    model = SPCENCCFrontend(
        spcen      = spcen_to_script(frontend.spcen),
        dct        = frontend.dct,
        drop_first = frontend.drop_first
    )

    model.cpu()
    model.eval()
    
    #Script
    model = torch.jit.script(model)

    #Return
    return model