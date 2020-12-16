#! /usr/bin/python3
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Reshape,Dense,Dropout,Activation,Flatten, Convolution2D, MaxPooling2D, ZeroPadding2D, Permute
import tensorflow.keras.models as models
import tensorflow.keras as keras

import matplotlib.pyplot as plt

from build_dataset import build_dataset,get_num_elements_in_dataset, build_debug_set

import auto_encoder
import classifier

tf.random.set_seed(1337)

test_paths = [
    "/kek/Day_2_After_FFT/Device_9/tx_1/converted_576floats.protobin"
]

classifier_test_paths = [
    "/kek/Day_1_Equalized/Device_10/tx_10/converted_576floats.protobin",
    "/kek/Day_1_Equalized/Device_9/tx_10/converted_576floats.protobin", 
]

encoder_weights_path = "auto_encoder.wts.h5"
classifier_weights_path = "classifier.wts.h5"

the_encoder = auto_encoder.build_model()
the_encoder.load_weights(encoder_weights_path)

the_classifier = classifier.build_model()
the_classifier.load_weights(classifier_weights_path)


stack_input = keras.Input(shape=the_encoder.input.shape)
stack_layers = the_encoder(stack_input)
stack_layers = the_classifier(stack_layers)

the_stack = tf.keras.Model(inputs=stack_input, outputs=stack_layers)
the_stack.summary()

classifier.test_model(the_stack, test_paths)

classifier.test_model(the_classifier, classifier_test_paths)