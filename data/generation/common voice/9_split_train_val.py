import os
import pandas as pd
from sklearn.model_selection import train_test_split

#PARAMETER
CSV_TRAIN = #ADD PATH
CSV_VAL   = #ADD PATH

#Checht Pfad
if os.path.exists(CSV_VAL):
    raise Exception("VALIDATION DATASET ALREADY EXISTS")

#LÄDT CSV
df = pd.read_csv(CSV_TRAIN)
train, val = train_test_split(df, test_size=4096)

#EXPORT
train.to_csv(CSV_TRAIN, index=False)
val.to_csv  (CSV_VAL,   index=False)