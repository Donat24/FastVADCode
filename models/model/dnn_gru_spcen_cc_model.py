from typing import Optional, Tuple
import torch
from torch import Tensor
from torch import nn
from models.frontends.algorithms.sliding_window_context import SlidingWindowContext
from .vad_lightningbase import VADLightningBase
from models.frontends.spcen_cc import SPCENCCFrontendTrain
from torchvision.ops import MLP

def get_activation_layer(act:str):
    if act.lower() == "relu":
        return nn.ReLU
    elif act.lower() == "leakyrelu":
        return nn.LeakyReLU
    else:
        raise NotImplementedError()

#TrainClass
class DNN_GRU_SPCEN_CC_ModelTrain(VADLightningBase):
    def __init__(self,
            frame_size       = None,
            sr               = None,
            n_cc             = 14,
            drop_first       = True,

            #DNN
            dropout          = 0.1,
            act              = "leakyrelu",
            dnn_frames       = 7,
            dnn_hidden_size  = [128, 128, 128, 48],
            
            #GRU
            gru_hidden_size  = 48,
            gru_num_layers   = 2,
            bidirectional    = False
            
        ) -> None:

        super().__init__()

       #MFCC
        self.frontend  = SPCENCCFrontendTrain(
            frame_size    = frame_size,
            sr            = sr,
            n_cc          = n_cc,
            drop_first    = drop_first,
        )

        #DNN
        self.sliding_window_context = SlidingWindowContext(num_frames = dnn_frames)
        self.mlp = MLP(
            in_channels      = (n_cc - 1) * dnn_frames if drop_first else n_cc * dnn_frames,
            hidden_channels  = dnn_hidden_size,
            norm_layer       = None,
            activation_layer = get_activation_layer(act),
            dropout          = dropout,
        )

        #GRU
        self.gru = nn.GRU(
            input_size    = dnn_hidden_size [-1],
            hidden_size   = gru_hidden_size,
            num_layers    = gru_num_layers,
            batch_first   = True,
            bidirectional = bidirectional
        )

        #Out
        self.mlp_out = MLP(
            in_channels     = gru_hidden_size * 2 if bidirectional else gru_hidden_size,
            hidden_channels = [gru_hidden_size, 1],
            norm_layer      = None
        )


    def forward(self, x : Tensor):
        out     = x
        out     = self.frontend(out)
        
        #Layer 1
        out,_   = self.sliding_window_context(out)
        out     = out.flatten(-2,-1)
        out     = self.mlp(out)

        #GRU
        out,_   = self.gru(out)
        
        #Out
        out     = self.mlp_out(out)
        
        out = out.flatten(-2)
        
        return out

#class DNN_GRU_MFCC_ExportModel(nn.Module):
#    def __init__(self,
#        frontend                = None,
#        n_context               = None,
#        mlp                     = None,
#        gru                     = None,
#        mlp_out                 = None
#    ) -> None:
#
#        super().__init__()
#
#        #Parts
#        self.frontend    = frontend
#        self.mlp         = mlp
#        self.gru         = gru
#        self.mlp_out     = mlp_out
#        
#        #Context
#        self.n_context     = n_context
#        self.feature_size  = self.frontend.mfcc.n_mfcc - 1 if self.frontend.drop_first else self.frontend.mfcc.n_mfcc
#        
#    def forward(self, x : Tensor, dnn_context : Optional[Tensor] = None, h_context : Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
#        
#        #Reshape
#        x = x.view((1, 1, -1))
#        
#        #MFCCs
#        mfccs = self.frontend(x)
#
#        #Layer 1
#        if dnn_context is None:
#            dnn_context = torch.zeros((1, self.n_context, self.feature_size), dtype=x.dtype, device=x.device)
#        dnn_context = torch.concat((dnn_context, mfccs), dim = -2 )[..., - self.n_context : , : ]
#        out = self.mlp(dnn_context.flatten(-2,-1).view(1,1,-1))
#
#        #GRU
#        out, h_context = self.gru(out, h_context)
#
#        #Out
#        out = self.mlp_out(out)
#
#        #Sigmoid
#        out = torch.sigmoid(out)
#        
#        #Return
#        return out.flatten(), dnn_context, h_context
#
#def to_script(model:DNN_GRU_MFCC_ModelTrain):
#    
#    #Model
#    model.cpu()
#    model.eval()
#    model.freeze()
#    
#    model = DNN_GRU_MFCC_ExportModel(
#        frontend    = model.frontend,
#        n_context   = model.sliding_window_context.num_frames,
#        mlp         = model.mlp,
#        gru         = model.gru,
#        mlp_out     = model.mlp_out
#    )
#    model.cpu()
#    model.eval()
#    
#    #Script
#    model = torch.jit.script(model)
#
#    #Return
#    return model
#