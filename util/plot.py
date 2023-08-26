import torch
import matplotlib.pyplot as plt
import librosa
import numpy as np
import pandas as pd
import math
from nnAudio import features
import torchaudio
from collections import OrderedDict

from .util import *

#Torch Tensor -> NP Array
#Librosa kann nicht mit Toch Tensoren arbeiten ....
def _get_np_array(tensor):
    
    #tensor.numpy()
    if hasattr (tensor,"numpy"):
        return tensor.numpy()
    
    return tensor

#Erstellt neue Figure
def _create_new_fig():
    fig = plt.figure()
    ax = plt.axes()
    return fig, ax

def plot_waveform(waveform, sr = None, x_in_sec = True, y_axis_0dbfs_scale = False, ax = None):

    #Fixt x_in_sec
    if sr is None:
        x_in_sec = False

    #Erzeugt neuen Plot
    if ax is None:
        fig, ax = _create_new_fig()

    #x-Achse
    x_axis = torch.arange(0, waveform.shape[-1], dtype=torch.float32)
    if x_in_sec:
        x_axis /= sr

    #y-Achse
    y = waveform

    ax.plot(x_axis,y, label="Waveform")

    #X und Y Labels
    if x_in_sec:
        ax.set_xlabel("Zeit in Sekunden")
    else:
        ax.set_xlabel("Samples")
    
    ax.set_ylabel("Amplitute")

    #Y Axe zwischen -1 und 1
    if y_axis_0dbfs_scale:
        ax.set_ylim((-1,1))

    ax.set_title("Waveform Plot")

def plot_waveform_with_voice(waveform, voice = None, sr = None, x_in_sec = True, ax = None, alpha_voice = 0.1,**kwargs):

    #Fixt x_in_sec
    if sr is None:
        x_in_sec = False

    #Erzeugt neuen Plot
    if ax is None:
        fig, ax = _create_new_fig()
    
    #Plot Tensor
    plot_waveform(waveform, sr = sr, x_in_sec = x_in_sec, ax = ax, **kwargs)

    #Plot
    if voice is not None:
        for start, end in get_parts(voice):

            if x_in_sec:
                start = librosa.samples_to_time(start, sr=sr)
                end   = librosa.samples_to_time(end, sr=sr)

            #Grüner Hintergrund
            ax.axvspan( xmin = start, xmax = end, alpha = alpha_voice, color="green", lw=3, label="Sprache")
    

def plot_model_prediction(x, y, sample_length, hop_length, context_length = 0, sr = None, x_in_sec = True, prediction = None, pred_treshold = 0, sigmoid_prediction = True, plot_model_out = True, decay = librosa.time_to_samples(times = 0.1, sr = 16000), ax = None, **kwargs):

    #Fixt x_in_sec
    if sr is None:
        x_in_sec = False

    #Erzeugt neuen Plot
    if ax is None:
        fig, ax = _create_new_fig()

    #X für Plot
    if len(x.shape) == 1:
        waveform = x
    elif len(x.shape) == 2:
        waveform = reverse_unfold(x, hop_length)
    else:
        raise Exception("BAD TENSOR SHAPE")
    
    #Fixt context_length
    if context_length:
        waveform = waveform[context_length:]

    #Y für Plot
    if y is None:
        voice = None
    elif len(y.shape) == 1 and y.size(-1) > waveform.size(-1) - sample_length:
        voice = y
    elif len(y.shape) == 1:
        voice = y_to_full_length(y, sample_length, hop_length)
    else:
        raise Exception("BAD Y TENSOR SHAPE")

    #Plot
    plot_waveform_with_voice(waveform, voice, sr = sr, x_in_sec = x_in_sec, ax=ax, alpha_voice = 0.3, **kwargs)
    ax.set_title("")

    #Plot für Model Prediction
    if prediction != None:

        if len(prediction.shape) == 1 and prediction.size(-1) > waveform.size(-1) - sample_length:
            pass
        elif len(prediction.shape) == 1:
            prediction = y_to_full_length(prediction, sample_length, hop_length)
        else:
            raise Exception("BAD PRED TENSOR SHAPE")

        #Erstellt Parts
        for start, end in get_parts_with_buffer( prediction, treshold=pred_treshold, decay = decay):
            
            if x_in_sec:
                start = librosa.samples_to_time(start, sr=sr)
                end   = librosa.samples_to_time(end,   sr=sr)
        
            ax.axvspan(
                xmin = start, xmax = end, alpha = 0.2, edgecolor = "red", facecolor=(1,1,1,1), hatch=r"///", lw=3, label = "Vorhersage")
        
        #Value
        if plot_model_out:

            axis_prediction = ax.twinx()
            if sigmoid_prediction:
                prediction = torch.sigmoid(prediction)
            
            x_axis = torch.arange(0, waveform.shape[-1], dtype=torch.float32)
            if x_in_sec:
                x_axis /= sr
            
            #Plot auf neue Achse
            axis_prediction.plot(x_axis,prediction, label = "Modell-Output", color="red")
            axis_prediction.set_ylim(0,1)
            axis_prediction.set_ylabel("Modell")

            forms1, labels1 = ax.get_legend_handles_labels()
            forms2, labels2 = axis_prediction.get_legend_handles_labels()
            legend_labels = OrderedDict(zip(labels1 + labels2, forms1 + forms2))
            axis_prediction.legend(legend_labels.values(), legend_labels.keys())
        
        else:
            forms1, labels1 = ax.get_legend_handles_labels()
            legend_labels = OrderedDict(zip(labels1, forms1))
            ax.legend(legend_labels.values(), legend_labels.keys())



def plot_batch(
        #Data
        x, y, sample_length, hop_length, context_length = 0,
        
        #Pred
        prediction = None, pred_treshold = 0,
        
        #Füllt Lücken
        decay = librosa.time_to_samples(times = 0.1, sr = 16000),
        
        #Überschrift
        sample_idx = None,
        
        #Layout
        x_axis_plots = 4, subplot_width = 3, subplot_height = 1,
        
        #Plottet welches Bild gerade erstellt wird
        log_curr_img = False,
        
        #Other Args
        **kwargs
    ):

    #Batch IDX
    _x_size = x.size(0) if hasattr(x,"size") else len(x)
    if sample_idx is not None:
        if len(sample_idx) != _x_size:
            raise Exception("BAD SIZE OF SAMPLE_IDX")
    else:
        sample_idx = range(_x_size)

    
    #Plot Layout
    y_axis_plots = math.ceil( len(sample_idx) / x_axis_plots )

    #Erzeugt neuen Plot
    fig = plt.figure(figsize = (subplot_width * x_axis_plots, subplot_height * y_axis_plots))

    for counter, idx in enumerate(sample_idx):
        
        #Axis
        curr_x  = counter % x_axis_plots
        curr_y  = counter // x_axis_plots
        
        #Für größere Plots
        if log_curr_img:
            print(f"Erstelle Plot: IDX : '{idx}' | X : '{curr_x}' | Y : '{curr_y}'")
        
        curr_ax = plt.subplot2grid((y_axis_plots, x_axis_plots), (curr_y, curr_x), fig=fig)

        #Params for Plot
        plot_x    = x[counter]
        plot_y    = y[counter]
        plot_pred = prediction[counter] if prediction != None else None
        
        #Subplot
        plot_model_prediction(plot_x, plot_y, sample_length, hop_length, context_length = context_length, prediction = plot_pred, pred_treshold = pred_treshold, decay = decay, ax = curr_ax)
        curr_ax.set_title(f"Sample {idx}")
        curr_ax.set_xlabel("")
        curr_ax.set_ylabel("")
    
    #Plot
    fig.tight_layout()

def plot_model(
    #Data
    sample_idx, dataset, model_predict_func, pred_treshold = 0,

    #Fügt Verlängerung für Pred ein
    decay = librosa.time_to_samples(times = 0.1, sr = 16000),
    
    #Layout
    x_axis_plots = 4, subplot_width = 3, subplot_height = 1,

    #Other Args
    **kwargs
    ):

    x_list = []
    y_list = []
    pred_list = []

    #Lädt Samples
    with torch.no_grad():
        for idx in sample_idx:

            #Load Data
            x, y = dataset[idx]
            pred = model_predict_func(x)

            #Fügt Daten an
            x_list.append(x)
            y_list.append(y)
            pred_list.append(pred)
    
    #Plots Batch with Model Results
    return plot_batch(
        x_list, y_list, prediction = pred_list, pred_treshold = pred_treshold, decay = decay, sample_idx = sample_idx,
        sample_length = dataset.sample_length, hop_length = dataset.hop_length, context_length = dataset.context_length,
        x_axis_plots=x_axis_plots, subplot_width=subplot_width, subplot_height=subplot_height
    )


def plot_spectorgram(waveform, sr, sample_length, hop_length, low_treshold = -80, n_mels = None):
    
    #Cast -> Float
    waveform = waveform.to(torch.float)

    #Werte
    stft = features.STFT(n_fft = sample_length, hop_length = hop_length, sr = sr, output_format = "Magnitude")
    data = amp_to_db(rescale_fft_magnitude_with_window(tensor = stft(waveform)[0], window = stft.window_mask), low_treshold)

    #Mel
    if n_mels is None:

        #Plot
        spec = librosa.display.specshow(data.numpy(), y_axis='log', sr=sr, hop_length=hop_length, x_axis='time')
    
    else:

        #Mel
        transformer = torchaudio.transforms.MelScale(n_mels = n_mels, sample_rate = sr, n_stft = data.size(0), norm = "slaney")
        data = transformer(data)

        #Plot
        spec = librosa.display.specshow(data.numpy(), y_axis='mel', sr=sr, hop_length=hop_length, x_axis='time')

    #Rest
    plt.title("Spectogramm")
    plt.colorbar(spec, format="%+2.f dB")