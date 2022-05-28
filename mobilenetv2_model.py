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

traindf = pd.read_csv("./train.csv", dtype=str)
# traindf["clip_name"]=traindf["clip_name"].apply(append_ext)
# traindf.head()

# LOADING IMAGES
train_path = "./train_val_split/train/"
validation_path = "./train_val_split/val/"
# test_path = "./test_2022/"

train_dir = os.path.join(train_path)
validation_dir = os.path.join(validation_path)
# test_dir = os.path.join(test_path)

BATCH_SIZE = 16
IMG_SIZE = (160, 160)

train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir,
                                                            shuffle=True,
                                                            batch_size=BATCH_SIZE,
                                                            image_size=IMG_SIZE)

validation_dataset = tf.keras.utils.image_dataset_from_directory(validation_dir,
                                                                 shuffle=True,
                                                                 batch_size=BATCH_SIZE,
                                                                 image_size=IMG_SIZE)

# CREATING A TEST SET FROM THE VALIDATION SET
val_batches = tf.data.experimental.cardinality(validation_dataset)
test_dataset = validation_dataset.take(val_batches // 5)
validation_dataset = validation_dataset.skip(val_batches // 5)

# BUFFERED PREFETCHING TO LOAD IMGS FROM DISK
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

# DATA AUGMENTATION FOR MFCC
# data_augmentation = tf.keras.Sequential([
#     # tf.keras.layers.RandomFlip('horizontal'),
#     # tf.keras.layers.RandomRotation(0.2),
#     tf.keras.layers.RandomWidth(factor=(0.2, 0.3), interpolation='gaussian')
# ])


# USING PREPROCESSING TO RESCALE PIXEL VALUES
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
rescale = tf.keras.layers.Rescaling(1. / 127.5, offset=-1)

# Create the base model from the pre-trained model MobileNet V2
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

# FEATURE EXTRACTOR CONVERTS EACH IMAGE INTO A 5*5*1280 BLOCK OF FEATURES
image_batch, label_batch = next(iter(train_dataset))
feature_batch = base_model(image_batch)
print(feature_batch.shape)

# freezing the convolutional base
base_model.trainable = False

# converting the features to a single 1280-element vector per image
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)

# converting features into a single prediction per image
prediction_layer = tf.keras.layers.Dense(1)
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)

# BUILDING THE MODEL
inputs = tf.keras.Input(shape=(160, 160, 3))
# x = data_augmentation(inputs)
# x = preprocess_input(x)
x = preprocess_input(inputs)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

# compiling the model
base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])


# TRAINING THE BASE MODEL
initial_epochs = 10

history = model.fit(train_dataset,
                    epochs=initial_epochs,
                    validation_data=validation_dataset)

# PLOTTING THE RESULTS
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='upper left')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()), 1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper left')
plt.ylabel('Cross Entropy')
plt.ylim([0, 1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

#fn = "9th_training_20epochs_freq_mask"
#plt.savefig(fn, format="png")
#print(f"Saving '{fn}.png'")

# EVALUATING THE MODEL
loss1, accuracy1 = model.evaluate(validation_dataset)

print("Validation loss: {:.2f}".format(loss1))
print("Validation accuracy: {:.2f}".format(accuracy1))

# # unfreezing the convolutional base
# base_model.trainable = True
#
# # fine tuning from the 100th layer
# fine_tune_at = 100
#
# # Freeze all the layers before the `fine_tune_at` layer
# for layer in base_model.layers[:fine_tune_at]:
#   layer.trainable = False
#
# # setting lower learning rate to reduce overfitting
# model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
#               optimizer = tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate/10),
#               metrics=['accuracy'])
#
# fine_tune_epochs = 10
# total_epochs = initial_epochs + fine_tune_epochs
#
# history_fine = model.fit(train_dataset,
#                          epochs=total_epochs,
#                          initial_epoch=history.epoch[-1],
#                          validation_data=validation_dataset)
#
# acc += history_fine.history['accuracy']
# val_acc += history_fine.history['val_accuracy']
#
# loss += history_fine.history['loss']
# val_loss += history_fine.history['val_loss']
#
# plt.figure(figsize=(8, 8))
# plt.subplot(2, 1, 1)
# plt.plot(acc, label='Training Accuracy')
# plt.plot(val_acc, label='Validation Accuracy')
# plt.ylim([0.6, 1])
# plt.plot([initial_epochs-1,initial_epochs-1],
#           plt.ylim(), label='Start Fine Tuning')
# plt.legend(loc='upper left')
# plt.title('Training and Validation Accuracy')
#
# plt.subplot(2, 1, 2)
# plt.plot(loss, label='Training Loss')
# plt.plot(val_loss, label='Validation Loss')
# plt.ylim([0, 1.0])
# plt.plot([initial_epochs-1,initial_epochs-1],
#          plt.ylim(), label='Start Fine Tuning')
# plt.legend(loc='upper left')
# plt.title('Training and Validation Loss')
# plt.xlabel('epoch')
# plt.show()
#
# fn = "9th_training_20epochs_fine_tuned_dropout20_freq_mask"
# plt.savefig(fn, format="png")
# print(f"Saving '{fn}.png'")
#
# # EVALUATING THE MODEL
# loss, accuracy = model.evaluate(test_dataset)
# print('Test loss:', loss)
# print('Test accuracy:', accuracy)

# # Retrieve images from the test set
# image_batch, label_batch = test_dataset.as_numpy_iterator().next()
# predictions = model.predict_on_batch(image_batch).flatten()
#
# # Apply a sigmoid since the model returns logits
# predictions = tf.nn.sigmoid(predictions)
# predictions = tf.where(predictions < 0.5, 0, 1)
#
# print('Predictions:\n', predictions.numpy())
# print('Labels:\n', label_batch)

