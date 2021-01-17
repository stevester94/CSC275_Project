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

#######################################################################################################
# FFT limited subset
# Populated
# ds_config = {
#     "size": 1800003,   
#     "cache_name": "day2.fft.limited",
#     "paths": [  
#         '/mnt/lebensraum/Datasets/Day1.Equalized/Device_10/tx_5/converted_576floats.protobin',
#         '/mnt/lebensraum/Datasets/Day1.Equalized/Device_10/tx_4/converted_576floats.protobin',
#         '/mnt/lebensraum/Datasets/Day2.After_FFT/Device_11/tx_3/converted_576floats.protobin',
#         '/mnt/lebensraum/Datasets/Day2.After_FFT/Device_11/tx_8/converted_576floats.protobin',
#         '/mnt/lebensraum/Datasets/Day2.After_FFT/Device_12/tx_10/converted_576floats.protobin',
#         '/mnt/lebensraum/Datasets/Day2.After_FFT/Device_12/tx_4/converted_576floats.protobin',
#         '/mnt/lebensraum/Datasets/Day2.After_FFT/Device_13/tx_1/converted_576floats.protobin',
#         '/mnt/lebensraum/Datasets/Day2.After_FFT/Device_13/tx_9/converted_576floats.protobin',
#     ],
# }

ds_config = {
    "size": 1451369,
    "cache_name": "day2.fft.dev9_only",
    "paths": [
        '/mnt/lebensraum/Datasets/Day2.After_FFT/Device_9/tx_7/converted_576floats.protobin',
        '/mnt/lebensraum/Datasets/Day2.After_FFT/Device_9/tx_9/converted_576floats.protobin',
        '/mnt/lebensraum/Datasets/Day2.After_FFT/Device_9/tx_5/converted_576floats.protobin',
        '/mnt/lebensraum/Datasets/Day2.After_FFT/Device_9/tx_3/converted_576floats.protobin',
    ],
}

encoder_weights_path = "/mnt/lebensraum/CSC275_Project/working_set/golden_weights/encoder.objective.wts.h5"
classifier_weights_path = "/mnt/lebensraum/CSC275_Project/working_set/golden_weights/classifier.objective.wts.h5"

the_encoder = auto_encoder.build_model()
the_encoder.load_weights(encoder_weights_path)

the_classifier = classifier.build_model()
the_classifier.load_weights(classifier_weights_path)


stack_input = keras.Input(shape=the_encoder.input.shape)
stack_layers = the_encoder(stack_input)
stack_layers = the_classifier(stack_layers)

the_stack = tf.keras.Model(inputs=stack_input, outputs=stack_layers)
the_stack.summary()

classifier.test_model(the_stack, ds_config)

# classifier.test_model(the_classifier, classifier_test_paths)