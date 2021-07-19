import xenocanto
import sys
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.backend import relu
from tensorflow.python.keras.layers.convolutional import Convolution2D
from tensorflow.python.keras.layers.core import Dropout
from tensorflow.python.keras.layers.pooling import MaxPool2D
from tensorflow.python.util.nest import flatten


TEST_SIZE = 0.4
EPOCHS = 10

def main():
    # load 
    sounds, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(sounds), np.array(labels), test_size=TEST_SIZE
    )

    # get compiled neural network
    model = get_model()

    # train model using training sets
    model.fit(x_train, y_train, epochs=EPOCHS)

    # test model with test set
    model.evaluate(x_test, y_test, verbose=2)

def load_data(directory):
    # loads directory
    # assumes directory contains sound files of bird songs organized into folders by bird type (category)
    # returns tuple ('sounds', 'label'). audio should be segmented and edited.

    raise NotImplementedError


def get_model():
    # returns a compiled convolutional neural network. output layer should have 1-NUM_CATEGORIES units,
    # one for each category.

    raise NotImplementedError

if __name__ == "__main__":
    main()