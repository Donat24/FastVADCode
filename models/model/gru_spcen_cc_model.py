from typing import Optional, Tuple
import torch
from torch import Tensor
from torch import nn
from models.frontends.algorithms.sliding_window_context import SlidingWindowContext
from .vad_lightningbase import VADLightningBase
from models.frontends.spcen_cc import SPCENCCFrontendTrain
from models.frontends.spcen_cc import to_script as spcencc_to_script
from torchvision.ops import MLP

def get_activation_layer(act:str):
    if act.lower() == "relu":
        return nn.ReLU
    elif act.lower() == "leakyrelu":
        return nn.LeakyReLU
    else:
        raise NotImplementedError()

#TrainClass
class FastVad_Train(VADLightningBase):
    def __init__(self,
            frame_size       = None,
            sr               = None,
            n_cc             = 14,
            drop_first       = True,

            #DNN
            act              = "leakyrelu",
            hidden_size      = 48,
            gru_num_layers   = 2,
            
        ) -> None:

        super().__init__()

       #MFCC
        self.frontend  = SPCENCCFrontendTrain(
            frame_size    = frame_size,
            sr            = sr,
            n_cc          = n_cc,
            drop_first    = drop_first,
        )

        #In MLP
        self.in_mlp = nn.Sequential(
            nn.Linear(in_features=n_cc - 1 if drop_first else n_cc, out_features=hidden_size),
            get_activation_layer(act)()
        )
            
        #GRU
        self.gru = nn.GRU(
                input_size    = hidden_size,
                hidden_size   = hidden_size,
                num_layers    = gru_num_layers,
                batch_first   = True,
                bidirectional = False
        )

        #Out MLP
        self.out_mlp = MLP(
            in_channels      = hidden_size,
            hidden_channels  = [hidden_size, 1],
            activation_layer = get_activation_layer(act),
            norm_layer       = None
        )


    def forward(self, x : Tensor):
        out   = x
        out   = self.frontend(out)
        out   = self.in_mlp(out)
        out,_ = self.gru(out)
        out   = self.out_mlp(out)
        out   = out.flatten(-2)
        
        return out

class FastVad(nn.Module):
    def __init__(self,
        frontend                = None,
        in_mlp                  = None,
        gru                     = None,
        out_mlp                 = None
    ) -> None:

        super().__init__()

        #Parts
        self.frontend    = frontend
        self.in_mlp      = in_mlp
        self.gru         = gru
        self.out_mlp     = out_mlp
        
    def forward(self, x : Tensor, m_last_frame : Optional[Tensor], h_context : Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
        
        #Reshape
        out = x.view((1, 1, -1))
        out, m_last_frame = self.frontend(out, m_last_frame)
        out               = self.in_mlp(out)
        out, h_context    = self.gru(out, h_context)
        out               = self.out_mlp(out)
        out               = out.flatten(-2)
        out               = torch.sigmoid(out)
        
        #Return
        return out.flatten(), m_last_frame, h_context

def to_script(model:FastVad_Train):
    
    #Model
    model.cpu()
    model.eval()
    
    model = FastVad(
        frontend    = spcencc_to_script(model.frontend),
        in_mlp      = model.in_mlp,
        gru         = model.gru,
        out_mlp     = model.out_mlp
    )

    model.cpu()
    model.eval()
    
    #Script
    model = torch.jit.script(model)

    #Return
    return model