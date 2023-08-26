from typing import Optional, Tuple
import torch
import torchaudio
from torch import nn
from torch import Tensor
from torch.nn import functional as F

class SlidingWindowContext(torch.jit.ScriptModule):
    def __init__(self, num_frames) -> None:
        super().__init__()
        self.num_frames = num_frames
    
    def forward(self, x : Tensor, context_frames : Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        
        #BATCH TIME FEATURES
        if len(x.shape) != 3:
            raise Exception("BAD TENSOR SHAPE")
        
        #Output
        out = torch.zeros(
            size   = ( x.size(0), x.size(1), self.num_frames, x.size(2)),
            dtype  = x.dtype,
            device = x.device
        )
        
        #Checkt Contextframe
        if context_frames is None:
            context_frames = torch.zeros(
                size   =  (x.size(0), self.num_frames - 1, x.size(2)),
                dtype  = x.dtype,
                device = x.device
            )
        
        if context_frames.size(1) < self.num_frames - 1:
            raise Exception(f"CONTEXT-FRAME ZU KURZ, context-Frame-Größe: {context_frames.size(1)}")
        
        #Schneidet Context Frame zu
        context_frames = context_frames[:,- self.num_frames + 1 : , : ]

        #Fügt Context an
        curr_frames = x
        curr_frames = torch.cat((context_frames, curr_frames), dim = 1)

        #Iterriert Frames
        for idx in range(curr_frames.size(1) + 1 - self.num_frames):
            new_frame = curr_frames[:, idx : idx + self.num_frames, : ]
            new_frame = new_frame.unsqueeze(1)
        
            #New Spec
            out[ : , [idx], : ,:] = new_frame

        #Return
        context_frames = curr_frames[ : , - self.num_frames : , : ]
        return out, context_frames