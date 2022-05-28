import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os, sys
os.environ['LIBROSA_CACHE_DIR'] = '/tmp/'
#from specAugment import spec_augment_tensorflow
import tensorflow as tf
#import spec_augment_tensorflow_upgraded as spectaug
import librosa
import librosa.display
import tensorflow as tf
#from tensorflow_addons.image import sparse_image_warp
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
#import tensorflow_addons as tfa
from sparse_image_warp import sparse_image_warp
from augment import SpecAugment


def sparse_warp(spectrogram, time_warping_para=8):
    """Spec augmentation Calculation Function.

    'SpecAugment' have 3 steps for audio data augmentation.
    first step is time warping using Tensorflow's image_sparse_warp function.
    Second step is frequency masking, last step is time masking.

    # Arguments:
      mel_spectrogram(numpy array): audio file path of you want to warping and masking.
      time_warping_para(float): Augmentation parameter, "time warp parameter W".
        If none, default = 80 for LibriSpeech.

    # Returns
      mel_spectrogram(numpy array): warped and masked mel spectrogram.
    """

    fbank_size = tf.shape(input=spectrogram)
    n, v = fbank_size[1], fbank_size[2]

    # Step 1 : Time warping
    # Image warping control point setting.
    # Source
    pt = tf.random.uniform([], time_warping_para, n-time_warping_para, tf.int32) # radnom point along the time axis
    src_ctr_pt_freq = tf.range(v // 2)  # control points on freq-axis
    src_ctr_pt_time = tf.ones_like(src_ctr_pt_freq) * pt  # control points on time-axis
    src_ctr_pts = tf.stack((src_ctr_pt_time, src_ctr_pt_freq), -1)
    src_ctr_pts = tf.cast(src_ctr_pts, dtype=tf.float32)

    # Destination
    w = tf.random.uniform([], -time_warping_para, time_warping_para, tf.int32)  # distance
    dest_ctr_pt_freq = src_ctr_pt_freq
    dest_ctr_pt_time = src_ctr_pt_time + w
    dest_ctr_pts = tf.stack((dest_ctr_pt_time, dest_ctr_pt_freq), -1)
    dest_ctr_pts = tf.cast(dest_ctr_pts, dtype=tf.float32)

    # warp
    source_control_point_locations = tf.expand_dims(src_ctr_pts, 0)  # (1, v//2, 2)
    dest_control_point_locations = tf.expand_dims(dest_ctr_pts, 0)  # (1, v//2, 2)

    warped_image, _ = sparse_image_warp(spectrogram, source_control_point_locations, dest_control_point_locations)

    return warped_image


def frequency_masking(spectrogram, v, frequency_masking_para=7, frequency_mask_num=1):
    """Spec augmentation Calculation Function.

    'SpecAugment' have 3 steps for audio data augmentation.
    first step is time warping using Tensorflow's image_sparse_warp function.
    Second step is frequency masking, last step is time masking.

    # Arguments:
      mel_spectrogram(numpy array): audio file path of you want to warping and masking.
      frequency_masking_para(float): Augmentation parameter, "frequency mask parameter F"
        If none, default = 100 for LibriSpeech.
      frequency_mask_num(float): number of frequency masking lines, "m_F".
        If none, default = 1 for LibriSpeech.

    # Returns
      mel_spectrogram(numpy array): warped and masked mel spectrogram.
    """
    # Step 2 : Frequency masking
    fbank_size = tf.shape(input=spectrogram)
    n, v = fbank_size[1], fbank_size[2]

    for i in range(frequency_mask_num):
        f = tf.random.uniform([], minval=0, maxval=frequency_masking_para, dtype=tf.int32)
        v = tf.cast(v, dtype=tf.int32)
        f0 = tf.random.uniform([], minval=0, maxval=v-f, dtype=tf.int32)

        # warped_mel_spectrogram[f0:f0 + f, :] = 0
        mask = tf.concat((tf.ones(shape=(1, n, v - f0 - f, 1)),
                          tf.zeros(shape=(1, n, f, 1)),
                          tf.ones(shape=(1, n, f0, 1)),
                          ), 2)
        spectrogram = spectrogram * mask
    return tf.cast(spectrogram, dtype=tf.float32)


def time_masking(spectrogram, tau, time_masking_para=5, time_mask_num=1):
    """Spec augmentation Calculation Function.

    'SpecAugment' have 3 steps for audio data augmentation.
    first step is time warping using Tensorflow's image_sparse_warp function.
    Second step is frequency masking, last step is time masking.

    # Arguments:
      mel_spectrogram(numpy array): audio file path of you want to warping and masking.
      time_masking_para(float): Augmentation parameter, "time mask parameter T"
        If none, default = 27 for LibriSpeech.
      time_mask_num(float): number of time masking lines, "m_T".
        If none, default = 1 for LibriSpeech.

    # Returns
      mel_spectrogram(numpy array): warped and masked mel spectrogram.
    """
    fbank_size = tf.shape(input=spectrogram)
    n, v = fbank_size[1], fbank_size[2]

    # Step 3 : Time masking
    for i in range(time_mask_num):
        t = tf.random.uniform([], minval=0, maxval=time_masking_para, dtype=tf.int32)
        t0 = tf.random.uniform([], minval=0, maxval=tau-t, dtype=tf.int32)

        # mel_spectrogram[:, t0:t0 + t] = 0
        mask = tf.concat((tf.ones(shape=(1, n-t0-t, v, 1)),
                          tf.zeros(shape=(1, t, v, 1)),
                          tf.ones(shape=(1, t0, v, 1)),
                          ), 1)
        spectrogram = spectrogram * mask
    return tf.cast(spectrogram, dtype=tf.float32)


def spec_augment(spectrogram):

    v = spectrogram.shape[0]
    tau = spectrogram.shape[1]

    #warped_spectrogram = sparse_warp(spectrogram)

    warped_frequency_spectrogram = frequency_masking(spectrogram, v=v)

    warped_frequency_time_spectrogram = time_masking(warped_frequency_spectrogram, tau=tau)

    return warped_frequency_time_spectrogram

def visualization_tensor_spectrogram(spectrogram):
    """visualizing first one result of SpecAugment

    # Arguments:
      mel_spectrogram(ndarray): mel_spectrogram to visualize.
      title(String): plot figure's title
    """

    # Show mel-spectrogram using librosa's specshow.
    #plt.figure(figsize=(7, 5))
    fig = plt.figure(figsize=(7, 5), dpi=1000, frameon=False)
    ax = fig.add_axes([0,0,1,1], frameon=False)
    ax.axis('off')
    librosa.display.specshow(librosa.amplitude_to_db(spectrogram[0, :, :, 0], ref=np.max), y_axis='hz', fmax=8000, x_axis='time')
    #plt.colorbar(format='%+2.0f dB')
    #plt.title(title)
    #plt.tight_layout()
    plt.show()

# GENERATING REGULAR SPECTROGRAMS
audio_fpath = "./train_sounds/"
audio_clips = os.listdir(audio_fpath)
FIG_SIZE = (7, 5)

def generate_spectrogram():

    for i in audio_clips:
        spectrograms_path = "./aug_train_spects/"
        save_name = spectrograms_path + i + ".jpg"

        if not os.path.exists(save_name):
        # check if a file already exists
            signal, sample_rate = librosa.load(audio_fpath + i, sr=2000)

            hop_length = 128 # in num. of samples
            n_fft = 2048 # window in num. of samples
            stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)

            # calculate abs values on complex numbers to get magnitude
            spectrogram = np.abs(stft)

            # reshape spectrogram shape to [batch_size, time, frequency, 1]
            # shape = spectrogram.shape
            # spectrogram = np.reshape(spectrogram, (-1, shape[0], shape[1], 1))

            apply = SpecAugment(spectrogram, 'SM')
            time_warped = apply.time_warp()

            # Show time warped & masked spectrogram
            visualization_tensor_spectrogram(spectrogram=spec_augment(time_warped))

            plt.savefig(save_name, pil_kwargs={'quality': 95}, bbox_inches=0, pad_inches=0)
            plt.close()
            librosa.cache.clear()

# Creating sprectrograms for both train and test batch
# for i in audio_clips:
#     spectrograms_path = "./aug_train_spects/"
#     save_name = spectrograms_path + i + ".jpg"
#     # check if a file already exists
#     if not os.path.exists(save_name):
#         signal, sample_rate = librosa.load(audio_fpath + i, sr=2000)
#         generate_spectrogram(signal, sample_rate, save_name)
#         plt.close()
generate_spectrogram()