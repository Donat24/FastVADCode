import torch
import torchaudio
from torch import nn
from torch.nn import functional as F

class BinaryAccuracy(nn.Module):
    def __init__(self, threshold = 0) -> None:
        #Init
        super().__init__()
        
        #Params
        self.threshold = threshold
    
    def forward(self, pred, y):
        
        #Flatten
        pred = pred.flatten()
        y    = y.flatten()

        #Size
        if pred.size(0) != y.size(0):
            raise Exception("TENSOR-SHAPES DOESNT MATCH")

        #shiftet Threshold auf 0
        if self.threshold:
            pred = pred - self.threshold

        #Heavyside sorgt für 0 oder 1
        pred = torch.heaviside(pred, values=torch.tensor(1,dtype= pred.dtype))
        
        #Return
        return (pred == y).sum() / y.size(0)