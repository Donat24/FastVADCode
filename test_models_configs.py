from train import train_model
import models
from util.datasets import *
from data.data_config import *
from data.dataloader import dataloader_train, dataloader_val

CONFIGS = [

    {
        "model_cls" : models.FastVad_Train,
        "params"    : {
            "frame_size"       : SAMPLE_LENGTH,
            "sr"               : SAMPLE_RATE,
            "n_cc"             : 20,
            "hidden_size"      : 48,
            "gru_num_layers"   : 2,
        }
    },
]


for config in CONFIGS:

    #Anzhl an Modellen
    numb_models = 1
    if "NUMB_MODELS" in config.keys():
        numb_models = config.pop("NUMB_MODELS")

    for run in range(numb_models):
        
        try:
                        
            model = config["model_cls"](**config["params"])
            name_params = config["params"]

            #Train
            train_model(
                model       = model,
                max_epochs  = 3,
                name_params = name_params,
            )
        except Exception as e:
            print(e)