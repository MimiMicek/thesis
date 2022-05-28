import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['LIBROSA_CACHE_DIR'] = '/tmp/'
from augment import SpecAugment


# Unzipping the sound files from train & test folders
# shutil.unpack_archive("train.zip", "")
# shutil.unpack_archive("test.zip", "")


# GENERATING REGULAR SPECTROGRAMS
audio_fpath = "./train_sounds/"
#audio_fpath = "./data/test/"
audio_clips = os.listdir(audio_fpath)
FIG_SIZE = (7, 5)

def generate_spectrogram(signal, sample_rate, save_name):

    hop_length = 128 # in num. of samples
    n_fft = 2048 # window in num. of samples

    # creating MFCC spectrograms
    mfcc = librosa.feature.mfcc(signal, n_fft=n_fft, hop_length=hop_length, n_mfcc=39)

    # plotting the spectrogram
    fig = plt.figure(figsize=FIG_SIZE, dpi=1000, frameon=True)
    ax = fig.add_axes([0, 0, 1, 1], frameon=True)
    ax.axis('on')
    img = librosa.display.specshow(mfcc, x_axis='time', hop_length=hop_length, sr=2000, vmin=-500, vmax=500)
    plt.colorbar()
    #plt.tight_layout()
    plt.ylabel("MFCC coefficients")
    plt.savefig(save_name, pil_kwargs={'quality': 95}, bbox_inches=0, pad_inches=0)
    librosa.cache.clear()

# Creating sprectrograms for both train and test batch
for i in audio_clips:
    spectrograms_path = "./aug_train_spects/"
    #spectrograms_path = "./test_spectrograms/"
    save_name = spectrograms_path + i + ".jpg" # i[:-5] without the .aiff
    # check if a file already exists
    if not os.path.exists(save_name):
        signal, sample_rate = librosa.load(audio_fpath + i, sr=2000)
        generate_spectrogram(signal, sample_rate, save_name)
        plt.close()
