import torch
from torch import nn
from torch.nn import functional as F
import torchmetrics
from torchmetrics.classification import BinaryF1Score
import lightning.pytorch as pl
from lion_pytorch import Lion

from util.util import *
import util.metric as metric
from util.exception import TrainException
from data.data_config import SAMPLE_LENGTH, HOP_LENGTH

#Für einfache Netzte ohne Puffer
class VADLightningBase(pl.LightningModule):
    def __init__(self) -> None:        
        
        super().__init__()        
        
        #Metriken
        self.loss_fn         = F.binary_cross_entropy_with_logits
        self.accuracy        = metric.BinaryAccuracy( threshold = 0)
        self.f1_score        = BinaryF1Score( threshold = 0.5 )

    def training_step(self, batch, batch_idx):

        #Batch
        x, y = batch

        #Out
        output = self(x)
        
        #Loss
        loss = self.loss_fn(output, y)
        
        self.log("train_loss", loss)
        return loss
    
    #Erzeugt Tensor der wie Y aussieht
    def forward_whole_file(self, x):

        #Tensor Shape
        if len(x.shape) != 2:
            raise Exception("BAD TENSOR SHAPE")
        
        #Out
        out = self(x.unsqueeze(0)).squeeze(0)

        #Return
        return y_to_full_length(out, SAMPLE_LENGTH, HOP_LENGTH)

    def test_step(self, batch, batch_idx = None):
        
        with torch.no_grad():

            x, y = batch
            
            #Out
            output = self.forward_whole_file(x)
            output = output[..., : y.size(-1)]
            
            #Metrics
            loss    = self.loss_fn(output, y)
            acc     = self.accuracy(output, y)
            f1      = self.f1_score(torch.sigmoid(output), y)

            #log
            if self._trainer is not None:
                self.log("test_loss", loss)
                self.log("test_acc",  acc)
                self.log("test_f1",   f1)

            return { "loss" : loss, "acc" : acc , "f1" : f1}
    
    def validation_step(self, batch, batch_idx = None):
        
        with torch.no_grad():
            
            x, y, = batch

            #Out
            output = self(x)

            #Metrics
            loss    = self.loss_fn(output, y)
            acc     = self.accuracy(output, y)
            f1      = self.f1_score(torch.sigmoid(output), y)

            #Debug
            if torch.isnan(loss):
                raise TrainException(
                    model = self,
                    batch = x,
                    out   = output,
                    loss  = loss,
                )

            self.log("val_loss", loss)
            self.log("val_acc",  acc )
            self.log("val_f1",   f1)
            
            return { "loss" : loss, "acc" : acc , "f1" : f1}

    def configure_optimizers(self):        
        optimizer = Lion(self.parameters(), lr = 1e-4)
        return optimizer