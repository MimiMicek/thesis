import librosa
import argparse
import numpy as np
import librosa.display
from augment import SpecAugment
import matplotlib.pyplot as plt
import os
#
# # Arguments
# parser = argparse.ArgumentParser()
# parser.add_argument('--dir', default='./LibriSpeech/', help='path to dataset/dir to look for files')
# parser.add_argument('--policy', default='LD', help='augmentation policies - LB, LD, SM, SS')
#
# args = parser.parse_args()

audio_fpath = "./train_sounds/"
audio_clips = os.listdir(audio_fpath)

# if __name__ == '__main__':


# make a list of all training files in the LibriSpeech Dataset
training_files = librosa.util.find_files(audio_fpath, ext=['aiff'], recurse=True)
print('Number of Training Files: ', len(training_files))

# Loop over files and apply SpecAugment
for file in training_files:

    # Load the audio file
    audio, sr = librosa.load(file)

    # Extract Mel Spectrogram Features from the audio file
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=256, hop_length=128, fmax=8000)
    plt.figure(figsize=(5, 3))
    librosa.display.specshow(librosa.power_to_db(mel_spectrogram, ref=np.max), x_axis='time', y_axis='mel', fmax=8000) # Base

    # Apply SpecAugment
    apply = SpecAugment(mel_spectrogram, 'SM')

    time_warped = apply.time_warp() # Applies Time Warping to the mel spectrogram
    #plt.figure(figsize=(14, 6))
    #librosa.display.specshow(librosa.power_to_db(time_warped[0, :, :, 0].numpy(), ref=np.max), x_axis='time', y_axis='mel', fmax=8000) # Time Warped

    freq_masked = apply.freq_mask() # Applies Frequency Masking to the mel spectrogram

    time_masked = apply.time_mask() # Applies Time Masking to the mel spectrogram
    plt.figure(figsize=(5, 3))
    librosa.display.specshow(librosa.power_to_db(time_masked[0, :, :, 0], ref=np.max), x_axis='time', y_axis='mel', fmax=8000) # Time Masked

    spectrograms_path = "./aug_train_spects/"
    save_name = spectrograms_path + file + ".jpg"
    plt.savefig(save_name, pil_kwargs={'quality': 95}, bbox_inches=0, pad_inches=0)


    # for i in audio_clips:
    #     spectrograms_path = "./aug_train_spects/"
    #     #spectrograms_path = "./test_spectrograms/"
    #     save_name = spectrograms_path + i + ".jpg" # i[:-5] without the .aiff
    #     # check if a file already exists
    #     if not os.path.exists(save_name):
    #         signal, sample_rate = librosa.load(audio_fpath + i, sr=2000)
    #         generate_spectrogram(signal, sample_rate, save_name)
    #         plt.close()