import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#Params
csv_in_path        = #ADD PATH
file_path          = #ADD PATH
csv_train_out_path = #ADD PATH
csv_test_out_path  = #ADD PATH

#liest CSV
ava = pd.read_csv(csv_in_path, header=None)

#Liest Datei
files = os.listdir(file_path)
ids   = [file.split(".")[0] for file in files]

def get_tuple(row):
    start = np.round(row[1] - 900, decimals = 2)
    end   = np.round(row[2] - 900, decimals = 2)
    label = row[3]
    return(label, start, end)

_res = []  
for id in ids:
    ava_parts   = ava[ava[0] == id]
    audio_parts = ava_parts.apply(get_tuple ,axis=1).to_list()
    _res.append({
        "filename" : id + ".wav",
        "parts"    : audio_parts
    })

#To Frame
res = pd.DataFrame(_res)

#Split
train, test = train_test_split(res, test_size=10)

#Export
train.to_csv(csv_train_out_path, index=False)
test.to_csv(csv_test_out_path,   index=False)