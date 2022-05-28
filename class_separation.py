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

# READING THE LABELS
# def append_ext(fn):
#     return fn + ".jpg"

traindf=pd.read_csv("./train.csv", dtype=str)
# traindf["clip_name"]=traindf["clip_name"].apply(append_ext)
# traindf.head()

# SEPARATE IMAGES INTO CLASSES
train_dir = "./train_spectrograms"
# test_dir = "./test_2022"
# creating separating directory
classes = "./train/"
# classes = "./test/"

# if the folder does not exist create it
if not os.path.exists(classes):
    os.mkdir(classes)

for filename, class_name in traindf.values:
    # Create subdirectory with `class_name`
    if not os.path.exists(classes + str(class_name)):
        os.mkdir(classes + str(class_name))
    src_path = train_dir + '/'+ filename + '.jpg'
    dst_path = classes + str(class_name) + '/' + filename + '.jpg'
    try:
        shutil.copy(src_path, dst_path)
        print("Sucessful")
    except IOError as e:
        print('Unable to copy file {} to {}'
              .format(src_path, dst_path))
    except:
        print('When try copy file {} to {}, unexpected error: {}'
              .format(src_path, dst_path, sys.exc_info()))
