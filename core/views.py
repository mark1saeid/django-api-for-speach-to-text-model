import numpy as np
from django.conf import settings
from django.core.files.storage import default_storage
from django.shortcuts import render
from keras.applications.imagenet_utils import decode_predictions
import cv2
from keras.preprocessing.image import img_to_array, load_img
from tensorflow.python.keras.backend import set_session
import tensorflow as tf

##
import os
import numpy as np
from scipy.fftpack import fft
from scipy.io import wavfile
from scipy import signal
from glob import glob
import re
import pandas as pd
import gc

from keras import optimizers, losses, activations, models
# Core layers and Convolutional layers that help train on image data
from keras.layers import Convolution2D, Dense, Input, Flatten, Dropout, MaxPooling2D, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import keras
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pydub import AudioSegment
from pydub.silence import split_on_silence
#method to aplly fft in voice 
def custom_fft(y, fs):
    T = 1.0 / fs
    N = y.shape[0]
    yf = fft(y)
    xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
    # FFT is simmetrical, so we take just the first half
    # FFT is also complex, to we take just the real part (abs)
    vals = 2.0/N * np.abs(yf[0:N//2])
    return xf, vals

def log_specgram(audio, sample_rate, window_size=20,

                 step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, times, spec = signal.spectrogram(audio,
                                    fs=sample_rate,
                                    window='hann',
                                    nperseg=nperseg,
                                    noverlap=noverlap,
                                    detrend=False)
    return freqs, times, np.log(spec.T.astype(np.float32) + eps)


def pad_audio(samples):
    if len(samples) >= 16000: return samples
    else: return np.pad(samples, pad_width=(16000 - len(samples), 0), mode='constant', constant_values=(0, 0))


label_label = ['bed' ,'bird', 'cat', 'dog' ,'down', 'eight' ,'five' ,'four', 'go' ,'happy',
 'house' ,'left', 'marvin' ,'nine' ,'no' ,'off' ,'on' ,'one' ,'right', 'seven',
 'sheila' ,'silence', 'six' ,'stop' ,'unknown', 'up' ,'yes']

def speech_to_text(filepath,model):
  text = ""
  sound_file = AudioSegment.from_wav(filepath)

  audio_chunks = split_on_silence(sound_file, 
    # must be silent for at least half a second
    # make it shorter if the pause is short, like 100-250ms
    min_silence_len=190,

    # consider it silen/content/sent3.wavt if quieter than -16 dBFS
    silence_thresh=-30
   )
  for i, chunk in enumerate(audio_chunks):
    samples = chunk.get_array_of_samples()
    samples = pad_audio(samples)
    samples = signal.resample(samples, int(16000 / 16000 * samples.shape[0]))
    _, _, specgram = log_specgram(samples, sample_rate=16000)
    imgs = []
    imgs.append(specgram)
    imgs = np.array(imgs)
    imgs = imgs.reshape(tuple(list(imgs.shape) + [1]))
    x = model.predict(imgs)
    predicts = np.argmax(x, axis=1)
    predicts = [label_label[p] for p in predicts]
    text= text +" " + predicts[0]
  return text 
  


def index(request):
    if request.method == "POST":
        #
        # Django image API
        #
        file = request.FILES["imageFile"]
        file_name = default_storage.save(file.name, file)
        file_url = default_storage.path(file_name)

        #
        # https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/load_img
        #

        #tf.enable_eager_execution()

        model = tf.keras.models.load_model('//home/marksaeid/Downloads/ModelApi(1)/ModelApi/ModelAPI/model.hdf5')


### preprocessing
        
###      
       

        #processed_image = lungModel.preprocess_input(image_batch.copy())

        #
        # get the predicted probabilities
        #
       # with settings.GRAPH1.as_default():
        set_session(settings.SESS)

 ### prediction       
        predictions = speech_to_text(file_url,model)


        return render(request, "index.html", {"predictions": predictions})

    else:
        return render(request, "index.html")
    
    return render(request, "index.html")