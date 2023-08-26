import os
import pandas as pd
import subprocess

#Download
url  = #ADD URL
path = #ADD PATH

#Already downloaded
downloaded = os.listdir(path)

#Lädt Date
f = open(file="video.txt", mode="r")
available = f.readlines()
available = [path.strip() for path in available]

#Downlaod files
for video in [v for v in available if v not in downloaded]:
    subprocess.call(["curl", url + video, "--output" , os.path.join(path, video)])