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
IMG_HEIGHT = 48
IMG_WIDTH = 64
NUM_CATEGORIES = 1

def main():
    # load 
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    labels = np.argmax(labels, axis=1)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # get compiled neural network
    model = get_model()

    # train model using training sets
    model.fit(x_train, y_train, epochs=EPOCHS)

    # test model with test set
    model.evaluate(x_test, y_test, verbose=2)

def load_data(directory):
    # loads directory
    # assumes directory contains spectrograms of bird songs organized into folders by bird type (category)
    # returns tuple ('specs', 'label'). images are resized.
    images = []
    labels = []
    for (root,dirs,files) in os.walk(directory, topdown=True):
        for file in files:
            file_img = cv2.imread(os.path.join(root, file), cv2.IMREAD_UNCHANGED)
            if file_img is not None:
                resized = cv2.resize(file_img, (IMG_WIDTH, IMG_HEIGHT))
                images.append(resized)
                labels.append(os.path.split(root)[1])
    result = (images, labels)
    return tuple(result)

def get_model():
    # returns a compiled convolutional neural network. output layer should have NUM_CATEGORIES units,
    # one for each category.
    model = tf.keras.Sequential([
        tf.keras.layers.Convolution2D(32, (3, 3), activation="relu", input_shape=(IMG_HEIGHT, IMG_WIDTH, 4)),
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
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="sigmoid")
    ])

    model.summary()
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model

if __name__ == "__main__":
    main()