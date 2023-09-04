from typing import Optional, Tuple
import torch
from torch import Tensor
from torch import nn
from models.frontends.algorithms.sliding_window_context import SlidingWindowContext
from .vad_lightningbase import VADLightningBase
from models.frontends.spcen_cc import SPCENCCFrontendTrain
from torchvision.ops import MLP
from models.frontends.spcen_cc import to_script as spcencc_to_script
import math

def get_activation_layer(act:str):
    if act.lower() == "relu":
        return nn.ReLU
    elif act.lower() == "leakyrelu":
        return nn.LeakyReLU
    else:
        raise NotImplementedError()

#TrainClass
class DNN_DNN_SPCEN_CC_ModelTrain(VADLightningBase):
    def __init__(self,
            frame_size              = None,
            sr                      = None,
            n_cc                    = 20,
            drop_first              = True,

            #DNN
            dropout                 = 0.1,
            act                     = "leakyrelu",

            #Layer 1
            frames_layer_1          = 7,
            hidden_size_layer_1     = [128, 128, 32],
            
            #Layer 2
            frames_layer_2          = 3 * 5 + 1,
            frames_hop_size_layer_2 = 3,
            hidden_size_layer_2     = [128, 32],
            
        ) -> None:

        super().__init__()

        #MFCC
        self.frontend  = SPCENCCFrontendTrain(
            frame_size    = frame_size,
            sr            = sr,
            n_cc          = n_cc,
            drop_first    = drop_first,
        )

        #Layer 1
        self.sliding_window_layer_1 = SlidingWindowContext(num_frames = frames_layer_1)

        self.mlp_layer_1 = MLP(
            in_channels      = (n_cc - 1) * frames_layer_1 if drop_first else n_cc * frames_layer_1,
            hidden_channels  = hidden_size_layer_1,
            norm_layer       = None,
            activation_layer = get_activation_layer(act),
            dropout          = dropout,
        )

        #Layer 2
        self.sliding_window_layer_2  = SlidingWindowContext(num_frames = frames_layer_2)
        self.frames_hop_size_layer_2 = frames_hop_size_layer_2

        self.mlp_layer_2 = MLP(
            in_channels      = math.ceil(frames_layer_2 / frames_hop_size_layer_2) * hidden_size_layer_1 [-1],
            hidden_channels  = hidden_size_layer_2,
            norm_layer       = None,
            activation_layer = get_activation_layer(act),
            dropout          = dropout,
        )

        #MLP
        self.mlp_out = MLP(
            in_channels     = hidden_size_layer_2[-1],
            hidden_channels = [hidden_size_layer_2[-1], 1],
            norm_layer      = None
        )


    def forward(self, x : Tensor):
        out     = x
        out     = self.frontend(out)
        
        #Layer 1
        out,_   = self.sliding_window_layer_1(out)
        out     = out.flatten(-2,-1)
        out     = self.mlp_layer_1(out)

        #Layer 2
        out,_   = self.sliding_window_layer_2(out)
        out     = out[...,::self.frames_hop_size_layer_2,:]
        out     = out.flatten(-2,-1)
        out     = self.mlp_layer_2(out)

        #Out
        out     = self.mlp_out(out)

        out = out.flatten(-2)
        
        return out

class DNN_DNN_SPCEN_CC_Model(nn.Module):
    def __init__(self,
        frontend                = None,
        n_layer_1_context       = None,
        n_layer_2_context       = None,
        layer_2_frames_hop_size = None,
        mlp_layer_1             = None,
        mlp_layer_2             = None,
        mlp_out                 = None
    ) -> None:

        super().__init__()

        #Parts
        self.frontend    = frontend
        self.mlp_layer_1 = mlp_layer_1
        self.mlp_layer_2 = mlp_layer_2
        self.mlp_out     = mlp_out
        
        #Context
        self.n_layer_1_context       = n_layer_1_context
        self.n_layer_1_feature_size  = self.frontend.dct.size(-1) - 1 if self.frontend.drop_first else self.frontend.dct.size(-1)
        self.n_layer_2_context       = n_layer_2_context
        self.layer_2_frames_hop_size = layer_2_frames_hop_size
        self.n_layer_2_feature_size  = self.mlp_layer_1[-2].out_features
        
    def forward(self, x : Tensor, m_last_frame : Optional[Tensor], layer_1_context : Optional[Tensor] = None, layer_2_context : Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        
        #Reshape
        out = x.view((1, 1, -1))
        
        #SPCEN CC
        out, m_last_frame = self.frontend(out, m_last_frame)

        #Layer 1
        if layer_1_context is None:
            layer_1_context = torch.zeros((1, self.n_layer_1_context, self.n_layer_1_feature_size), dtype=x.dtype, device=x.device)
        layer_1_context = torch.concat((layer_1_context, out), dim = -2 )[..., - self.n_layer_1_context : , : ]
        out = self.mlp_layer_1(layer_1_context.flatten(-2,-1).view(1,1,-1))

        #Layer 2
        if layer_2_context is None:
            layer_2_context = torch.zeros((1, self.n_layer_2_context, self.n_layer_2_feature_size), dtype=x.dtype, device=x.device)
        layer_2_context = torch.concat((layer_2_context, out), dim = -2 )[..., - self.n_layer_2_context : , : ]
        out = self.mlp_layer_2( layer_2_context[...,::self.layer_2_frames_hop_size,:].flatten(-2,-1).view(1,1,-1) )

        #Classifier
        out = self.mlp_out(out)

        #Sigmoid
        out = torch.sigmoid(out)
        
        #Return
        return out.flatten(), m_last_frame, layer_1_context, layer_2_context

def to_script(model:DNN_DNN_SPCEN_CC_ModelTrain):
    
    #Model
    model.cpu()
    model.eval()
    model.freeze()
    
    model = DNN_DNN_SPCEN_CC_Model(
        frontend = spcencc_to_script(model.frontend),
        n_layer_1_context = model.sliding_window_layer_1.num_frames,
        n_layer_2_context = model.sliding_window_layer_2.num_frames,
        layer_2_frames_hop_size = model.frames_hop_size_layer_2,
        mlp_layer_1 = model.mlp_layer_1,
        mlp_layer_2 = model.mlp_layer_2,
        mlp_out     = model.mlp_out
    )
    model.cpu()
    model.eval()
    
    #Script
    model = torch.jit.script(model)

    #Return
    return model
