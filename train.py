import argparse
import gc
import torchmetrics
import tqdm
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from fvcore.nn import FlopCountAnalysis
from tqdm import tqdm

import models
from util.util import *
from data.dataloader import *

#Torch Matmul
torch.set_float32_matmul_precision('high')

def train_model(model, max_epochs=1, max_steps = -1,limit_val_batches=1.0, accelerator = "auto", name_params = None,
        train_dataloaders = dataloader_train, val_dataloaders = dataloader_val,
    ):

    #Clean
    torch.cuda.empty_cache()
    gc.collect()

    #Model Name
    name = str( type(model).__name__ )
    if name_params:
        name = name + "_" + "_".join(f"{key}-{value}" for key, value in name_params.items())
        name = name.replace("[",".").replace("]",".").replace(",",".")

    trainer = pl.Trainer(

        #für Debugging
        accelerator = accelerator,

        #Training
        max_epochs = max_epochs,
        max_steps  = max_steps,

        #Logging
        logger=TensorBoardLogger("lightning_logs", name=name ),
        log_every_n_steps   = 100,
        val_check_interval  = 1000,
        limit_val_batches   = limit_val_batches,
        precision           = "16-mixed",
        gradient_clip_val   = 0.7,

        #Checkpoints
        callbacks=[
            ModelCheckpoint(
                monitor                 = "val_acc",
                mode                    = "max",
                save_top_k              = 2,
                save_last               = True,
                every_n_train_steps     = 1000,
                #every_n_epochs          = 1,
                save_on_train_epoch_end = True,
            )
        ]
        
    )

    trainer.fit(model=model, train_dataloaders = train_dataloaders, val_dataloaders = val_dataloaders)

def test_model(model, dataset = dataset_test):

    #Eval
    if model.training:
        model.eval()

    #Result
    result = []

    #Iter Dataset
    for idx, batch in tqdm(enumerate(dataset)):

        #cuda
        if "cuda" in str(model.device):
            batch = tuple([tensor.to(device = model.device) for tensor in batch])
        
        #Train-Step
        batch_result = model.test_step(batch)
        
        #Tensor -> item
        for key, value in batch_result.items():
            if torch.is_tensor(value):
                batch_result[key] = value.item()

        #IDx
        if "idx" not in batch_result:
            batch_result["idx"] = idx
        
        #Append
        result.append(batch_result)

    #Sort
    result.sort( key=lambda item: item["acc"])

    #Return
    return result