import pandas as pd
import os
import ast
import math

EXTRACTED_DIRECTORY = #ADD PATH
OUT_CSV_PATH        = #ADD PATH
OUT_CSV_SUFFIX      = #ADD PATH

#Liefert Zusammenhängende Parts
def get_parts (elements, duration, spacing = 100):
    duration = math.floor(duration * 1000)
    elements = [( max(element[0] - spacing, 0), min(element[1] + spacing, duration) ) for element in elements]
    
    _start = elements[0][0]
    _end   = elements[0][1]
    out = []
    for element in elements:
        if element[0] <= _end:
            _end = element[1]
        else:
            out.append((_start, _end))
            _start = element[0]
            _end   = element[1]
    out.append((_start, _end))
    return out

for language in os.listdir(EXTRACTED_DIRECTORY):

    #Main Dir
    language_dir = os.path.join(EXTRACTED_DIRECTORY, language)

    if not "transcription.csv" in os.listdir(language_dir):
        continue

    #Read CSV
    csv_path = os.path.join(language_dir, "transcription.csv")
    data = pd.read_csv(csv_path)
    
    
    data["wav2vec_result"] = data["wav2vec_result"].apply(ast.literal_eval)
    data["wav2vec_result"] = data["wav2vec_result"].apply(lambda element: list(zip(element["start_timestamps"],element["end_timestamps"])))
    data["parts"]          = data.apply(lambda row: get_parts(row["wav2vec_result"], row["duration"]), axis=1)
    data                   = data.drop(columns="wav2vec_result")
    data["filename"]       = data["path"]
    data                   = data.drop(columns="path")
    data["language"]       = language
    data.to_csv(os.path.join(OUT_CSV_PATH, language + "_" + OUT_CSV_SUFFIX), index=False)