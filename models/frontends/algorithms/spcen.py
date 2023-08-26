from typing import Optional
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

class SPCENTrain(nn.Module):
    def __init__(self, n_mels:int=1, s:float=0.0001, alpha:float = 0.98, delta:float = 2, r:float = 0.2 , eps:float = 1E-6) -> None:
        super().__init__()
        
        #Params
        self.s     = nn.Parameter( torch.log(torch.as_tensor( [ s     for i in range (n_mels) ] )),  requires_grad=True)
        self.alpha = nn.Parameter( torch.log(torch.as_tensor( [ alpha for i in range (n_mels) ] )),  requires_grad=True)
        self.delta = nn.Parameter( torch.log(torch.as_tensor( [ delta for i in range (n_mels) ] )),  requires_grad=True)
        self.r     = nn.Parameter( torch.log(torch.as_tensor( [ r     for i in range (n_mels) ] )),  requires_grad=True)
        self.eps   = eps
    
    def forward(self, x:Tensor, m_last_frame:Optional[Tensor] = None) -> Tensor:
        
        #Get Params
        s     = self.s.exp()
        alpha = self.alpha.exp()
        delta = self.delta.exp()
        r     = self.r.exp()
        eps   = self.eps

        
        out = torch.zeros_like(x)
        
        for idx in range(x.size(1)):
            frame = x[:, [idx] ,:]

            #Berechnet M
            if m_last_frame is None:
                m_last_frame = torch.zeros_like(frame,dtype = frame.dtype, device = frame.device)
            
            #IIR Filter
            m_curr_frame = ( (1 - s) * m_last_frame ) + ( s * frame )

            #Berechnet out
            out[: , [idx], : ] = (  ( (frame / (m_curr_frame + eps).pow(alpha) ) + delta).pow(r) - ( delta ** r ) ).to(dtype=out.dtype)

            #Setzt M neu
            m_last_frame = m_curr_frame

        
        return out


class SPCEN(nn.Module):
    def __init__(self, s:Tensor, alpha:Tensor, delta:Tensor, r:Tensor, eps:float = 1E-6) -> None:
        super().__init__()

        #Parameter
        self.s     = nn.Parameter(s     ,requires_grad=False)
        self.alpha = nn.Parameter(alpha ,requires_grad=False)
        self.delta = nn.Parameter(delta ,requires_grad=False)
        self.r     = nn.Parameter(r     ,requires_grad=False)
        self.eps   = eps
    
    def forward(self, frame:Tensor, m_last_frame:Optional[Tensor] = None) -> tuple[Tensor, Tensor]:

        #Berechnet M
        if m_last_frame is None:
            m_last_frame = torch.zeros_like(frame, dtype = frame.dtype, device = frame.device)
        
        #IIR Filter
        m_curr_frame = ( (1 - self.s).view(1,1,-1) * m_last_frame ).add_( self.s.view(1,1,-1) * frame )
        
        #Berechnet out
        #out = frame.div_( m_curr_frame.add_(self.eps).pow_(self.alpha) ).add_(self.delta).pow_(self.r).sub_(self.delta ** self.r)
        out = ( (frame / (m_curr_frame + self.eps).pow(self.alpha) ) + self.delta ).pow(self.r) - ( self.delta ** self.r )

        return out, m_curr_frame

def to_script(frontend:SPCENTrain):
    
    #Frontend
    frontend.cpu()
    frontend.eval()
    
    model = SPCEN(
        s     = frontend.s.exp(),
        alpha = frontend.alpha.exp(),
        delta = frontend.delta.exp(),
        r     = frontend.r.exp(),
        eps   = frontend.eps
    )
    
    model.cpu()
    model.eval()
    
    #Script
    model = torch.jit.script(model)

    #Return
    return model