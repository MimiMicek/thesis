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

#### CHANGE ALL PATHS

# train_path = "./train_2022/"
# train_folder = os.listdir(train_path)

# SPLIT TRAIN FOLDER TO TRAIN AND VALIDATION
splitfolders.ratio("train", output="train_val_split",
    seed=1337, ratio=(.8, .2), group_prefix=None, move=True)