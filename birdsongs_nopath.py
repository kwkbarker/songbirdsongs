import sys
import librosa
import librosa.display
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.backend import relu
from tensorflow.python.keras.layers.convolutional import Convolution2D
from tensorflow.python.keras.layers.core import Dropout
from tensorflow.python.keras.layers.pooling import MaxPool2D
from tensorflow.python.util.nest import flatten
from tensorflow import keras
import cv2
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FC


TEST_SIZE = 0.4
EPOCHS = 10
IMG_HEIGHT = 30
IMG_WIDTH = 30
NUM_CATEGORIES = 2

def main():
    # load 
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # get compiled neural network
    model = get_model()

    model.compile(
        optimizer=keras.optimizers.RMSprop(),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()]
    )

    # train model using training sets
    model.fit(x_train, y_train, epochs=EPOCHS)

    # test model with test set
    model.evaluate(x_test, y_test, verbose=2)

def load_data(directory):
    # loads directory
    # assumes directory contains sound files of bird songs organized into folders by bird type (category)
    # returns tuple ('sounds', 'label'). audio should be segmented and edited.
    images = []
    labels = []
    for (root,dirs,files) in os.walk(directory, topdown=True):
        for file in files:
            path = path = os.path.join(root, file)
            # file_img = audio_to_image(path)
            file_img = cv2.imread(os.path.join(path), cv2.IMREAD_UNCHANGED)
            if file_img is not None:
                # resized = cv2.resize(file_img, (IMG_WIDTH, IMG_HEIGHT))
                images.append(file_img)
                labels.append(os.path.split(root)[1])
    result = (images, labels)
    return tuple(result)

def audio_to_image(file):
    # converts audio file into spectrogram image.
    # assumes data is an audio file. returns corresponding spectrogram.

    clip, sr = librosa.load(file)
    S = librosa.stft(clip)
    S_db = librosa.amplitude_to_db(np.abs(S),ref=np.max)
    fig = plt.figure()
    canvas = FC(fig)
    librosa.display.specshow(S_db)
    canvas.draw()
    image = np.fromstring(canvas.tostring_rgb(), dtype='uint8')
    return image


def get_model():
    # returns a compiled convolutional neural network. output layer should have NUM_CATEGORIES units,
    # one for each category.
    model = tf.keras.Sequential([
        tf.keras.layers.Convolution2D(32, (3, 3), activation="relu", input_shape=(IMG_WIDTH,IMG_HEIGHT)),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        tf.keras.layers.Convolution2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPool2D(pool_size=(2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    return model

if __name__ == "__main__":
    main()