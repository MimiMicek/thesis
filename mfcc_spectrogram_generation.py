import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['LIBROSA_CACHE_DIR'] = '/tmp/'
import pandas as pd
import random
import scipy
from scipy import signal
from scipy.io import wavfile
import shutil
from shutil import copyfile
from sklearn.model_selection import train_test_split
import splitfolders
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, activations
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import regularizers, optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers

# Unzipping the sound files from train & test folders
# shutil.unpack_archive("train.zip", "")
# shutil.unpack_archive("test.zip", "")

# GENERATING MFCC SPECTROGRAMS
audio_fpath = "./train_sounds/"
#audio_fpath = "./test_sound/"
audio_clips = os.listdir(audio_fpath)
FIG_SIZE = (8, 6)

def generate_spectrogram(signal, sample_rate, save_name):

    hop_length = 128 # in num. of samples
    n_fft = 2048 # window in num. of samples
    #stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
    # calculate abs values on complex numbers to get magnitude
    #spectrogram = np.abs(stft)

    # creating MFCC spectrograms
    # mfcc = librosa.feature.mfcc(signal, n_fft=n_fft, hop_length=hop_length, n_mfcc=39)

    # plotting the spectrogram
    fig = plt.figure(figsize=FIG_SIZE, dpi=1000, frameon=False)
    ax = fig.add_axes([0,0,1,1], frameon=False)
    ax.axis('off')
    librosa.display.specshow(mfcc, sr=2000, hop_length=hop_length, x_axis='time', y_axis='hz')
    plt.savefig(save_name, pil_kwargs ={'quality': 95}, bbox_inches=0, pad_inches=0)
    librosa.cache.clear()

# Creating sprectrograms for both train and test batch
for i in audio_clips:
    spectrograms_path = "./train_mfccs/"
    #spectrograms_path = "./test_2022/"
    save_name = spectrograms_path + i + ".jpg" # i[:-5] without the .aiff
    # check if a file already exists
    if not os.path.exists(save_name):
        signal, sample_rate = librosa.load(audio_fpath + i,sr = 2000)
        generate_spectrogram(signal, sample_rate, save_name)
        plt.close()
