#! /usr/bin/python3
import sys, os

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Reshape,Dense,Dropout,Activation,Flatten, Convolution1D, MaxPooling2D, ZeroPadding2D, Permute
import tensorflow.keras.models as models
import tensorflow.keras as keras

import matplotlib.pyplot as plt

from build_dataset import build_dataset,get_num_elements_in_dataset, build_debug_set
import model_utils

tf.random.set_seed(1337)
MIN_CLASS_LABEL=9
MAX_CLASS_LABEL=13
NUM_CLASSES = MAX_CLASS_LABEL - MIN_CLASS_LABEL + 1
# whitelist = tf.constant([9, 10], dtype=tf.int64) # Which devices we want in our dataset

dr = 0.5 # dropout rate (%)
window_size=288

########################################################################################################
# Day 1 Equalized
# Accuracy = 0.905040823571175
# DATASET_SIZE=1126808
# train_paths = [
#     '/mnt/lebensraum/Datasets/Day1.Equalized/Device_9/tx_7/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day1.Equalized/Device_12/tx_3/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day1.Equalized/Device_12/tx_9/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day1.Equalized/Device_12/tx_6/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day1.Equalized/Device_12/tx_7/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day1.Equalized/Device_10/tx_5/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day1.Equalized/Device_11/tx_7/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day1.Equalized/Device_12/tx_10/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day1.Equalized/Device_9/tx_3/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day1.Equalized/Device_13/tx_7/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day1.Equalized/Device_10/tx_4/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day1.Equalized/Device_11/tx_2/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day1.Equalized/Device_10/tx_9/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day1.Equalized/Device_12/tx_2/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day1.Equalized/Device_10/tx_10/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day1.Equalized/Device_11/tx_8/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day1.Equalized/Device_10/tx_6/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day1.Equalized/Device_9/tx_6/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day1.Equalized/Device_11/tx_4/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day1.Equalized/Device_10/tx_8/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day1.Equalized/Device_12/tx_4/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day1.Equalized/Device_13/tx_9/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day1.Equalized/Device_12/tx_8/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day1.Equalized/Device_10/tx_7/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day1.Equalized/Device_13/tx_2/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day1.Equalized/Device_10/tx_3/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day1.Equalized/Device_9/tx_8/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day1.Equalized/Device_13/tx_3/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day1.Equalized/Device_11/tx_6/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day1.Equalized/Device_11/tx_5/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day1.Equalized/Device_13/tx_4/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day1.Equalized/Device_9/tx_1/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day1.Equalized/Device_10/tx_1/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day1.Equalized/Device_9/tx_4/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day1.Equalized/Device_11/tx_1/converted_576floats.protobin'
# ]

# test_paths = [
#     '/mnt/lebensraum/Datasets/Day1.Equalized/Device_11/tx_3/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day1.Equalized/Device_13/tx_10/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day1.Equalized/Device_12/tx_1/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day1.Equalized/Device_13/tx_5/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day1.Equalized/Device_9/tx_5/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day1.Equalized/Device_13/tx_6/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day1.Equalized/Device_13/tx_1/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day1.Equalized/Device_11/tx_10/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day1.Equalized/Device_9/tx_10/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day1.Equalized/Device_10/tx_2/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day1.Equalized/Device_9/tx_2/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day1.Equalized/Device_12/tx_5/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day1.Equalized/Device_13/tx_8/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day1.Equalized/Device_9/tx_9/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day1.Equalized/Device_11/tx_9/converted_576floats.protobin'
# ]

########################################################################################################
# Day 2 After FFT

# train_paths = [
#     '/mnt/lebensraum/Datasets/Day2.After_FFT/Device_9/tx_7/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day2.After_FFT/Device_13/tx_7/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day2.After_FFT/Device_13/tx_4/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day2.After_FFT/Device_9/tx_9/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day2.After_FFT/Device_11/tx_3/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day2.After_FFT/Device_13/tx_9/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day2.After_FFT/Device_13/tx_8/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day2.After_FFT/Device_12/tx_5/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day2.After_FFT/Device_13/tx_1/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day2.After_FFT/Device_9/tx_5/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day2.After_FFT/Device_13/tx_6/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day2.After_FFT/Device_12/tx_9/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day2.After_FFT/Device_11/tx_8/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day2.After_FFT/Device_11/tx_6/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day2.After_FFT/Device_12/tx_3/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day2.After_FFT/Device_12/tx_10/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day2.After_FFT/Device_11/tx_4/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day2.After_FFT/Device_12/tx_6/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day2.After_FFT/Device_12/tx_4/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day2.After_FFT/Device_11/tx_5/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day2.After_FFT/Device_9/tx_3/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day2.After_FFT/Device_9/tx_8/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day2.After_FFT/Device_10/tx_10/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day2.After_FFT/Device_10/tx_5/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day2.After_FFT/Device_10/tx_8/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day2.After_FFT/Device_11/tx_1/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day2.After_FFT/Device_9/tx_6/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day2.After_FFT/Device_10/tx_4/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day2.After_FFT/Device_11/tx_7/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day2.After_FFT/Device_13/tx_2/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day2.After_FFT/Device_12/tx_1/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day2.After_FFT/Device_9/tx_2/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day2.After_FFT/Device_11/tx_10/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day2.After_FFT/Device_10/tx_3/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day2.After_FFT/Device_10/tx_7/converted_576floats.protobin'
# ]

# test_paths = [
#     '/mnt/lebensraum/Datasets/Day2.After_FFT/Device_10/tx_6/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day2.After_FFT/Device_12/tx_7/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day2.After_FFT/Device_10/tx_9/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day2.After_FFT/Device_9/tx_10/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day2.After_FFT/Device_10/tx_2/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day2.After_FFT/Device_12/tx_2/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day2.After_FFT/Device_13/tx_10/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day2.After_FFT/Device_13/tx_5/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day2.After_FFT/Device_11/tx_2/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day2.After_FFT/Device_11/tx_9/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day2.After_FFT/Device_12/tx_8/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day2.After_FFT/Device_13/tx_3/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day2.After_FFT/Device_10/tx_1/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day2.After_FFT/Device_9/tx_4/converted_576floats.protobin',
#     '/mnt/lebensraum/Datasets/Day2.After_FFT/Device_9/tx_1/converted_576floats.protobin'
# ]

########################################################################################################
# Day 2 Equalized (For testing)
# Accuracy = 0.732852465997711 when used with Day 1 weights (vs )
# ds_config = {
#     "size":751428,
#     "cache_name": "day_2_equalized",
#     "train_paths":[],
#     "test_paths": [
#         "/mnt/lebensraum/Datasets/Day2.Equalized/Device_11/tx_6/converted_576floats.protobin",
#         "/mnt/lebensraum/Datasets/Day2.Equalized/Device_11/tx_3/converted_576floats.protobin",
#         "/mnt/lebensraum/Datasets/Day2.Equalized/Device_11/tx_1/converted_576floats.protobin",
#         "/mnt/lebensraum/Datasets/Day2.Equalized/Device_11/tx_10/converted_576floats.protobin",
#         "/mnt/lebensraum/Datasets/Day2.Equalized/Device_11/tx_7/converted_576floats.protobin",
#         "/mnt/lebensraum/Datasets/Day2.Equalized/Device_11/tx_4/converted_576floats.protobin",
#         "/mnt/lebensraum/Datasets/Day2.Equalized/Device_11/tx_2/converted_576floats.protobin",
#         "/mnt/lebensraum/Datasets/Day2.Equalized/Device_11/tx_5/converted_576floats.protobin",
#         "/mnt/lebensraum/Datasets/Day2.Equalized/Device_11/tx_8/converted_576floats.protobin",
#         "/mnt/lebensraum/Datasets/Day2.Equalized/Device_11/tx_9/converted_576floats.protobin",
#         "/mnt/lebensraum/Datasets/Day2.Equalized/Device_12/tx_6/converted_576floats.protobin",
#         "/mnt/lebensraum/Datasets/Day2.Equalized/Device_12/tx_3/converted_576floats.protobin",
#         "/mnt/lebensraum/Datasets/Day2.Equalized/Device_12/tx_1/converted_576floats.protobin",
#         "/mnt/lebensraum/Datasets/Day2.Equalized/Device_12/tx_10/converted_576floats.protobin",
#         "/mnt/lebensraum/Datasets/Day2.Equalized/Device_12/tx_7/converted_576floats.protobin",
#         "/mnt/lebensraum/Datasets/Day2.Equalized/Device_12/tx_4/converted_576floats.protobin",
#         "/mnt/lebensraum/Datasets/Day2.Equalized/Device_12/tx_2/converted_576floats.protobin",
#         "/mnt/lebensraum/Datasets/Day2.Equalized/Device_12/tx_5/converted_576floats.protobin",
#         "/mnt/lebensraum/Datasets/Day2.Equalized/Device_12/tx_8/converted_576floats.protobin",
#         "/mnt/lebensraum/Datasets/Day2.Equalized/Device_12/tx_9/converted_576floats.protobin",
#         "/mnt/lebensraum/Datasets/Day2.Equalized/Device_10/tx_6/converted_576floats.protobin",
#         "/mnt/lebensraum/Datasets/Day2.Equalized/Device_10/tx_3/converted_576floats.protobin",
#         "/mnt/lebensraum/Datasets/Day2.Equalized/Device_10/tx_1/converted_576floats.protobin",
#         "/mnt/lebensraum/Datasets/Day2.Equalized/Device_10/tx_10/converted_576floats.protobin",
#         "/mnt/lebensraum/Datasets/Day2.Equalized/Device_10/tx_7/converted_576floats.protobin",
#         "/mnt/lebensraum/Datasets/Day2.Equalized/Device_10/tx_4/converted_576floats.protobin",
#         "/mnt/lebensraum/Datasets/Day2.Equalized/Device_10/tx_2/converted_576floats.protobin",
#         "/mnt/lebensraum/Datasets/Day2.Equalized/Device_10/tx_5/converted_576floats.protobin",
#         "/mnt/lebensraum/Datasets/Day2.Equalized/Device_10/tx_8/converted_576floats.protobin",
#         "/mnt/lebensraum/Datasets/Day2.Equalized/Device_10/tx_9/converted_576floats.protobin",
#         "/mnt/lebensraum/Datasets/Day2.Equalized/Device_13/tx_6/converted_576floats.protobin",
#         "/mnt/lebensraum/Datasets/Day2.Equalized/Device_13/tx_3/converted_576floats.protobin",
#         "/mnt/lebensraum/Datasets/Day2.Equalized/Device_13/tx_1/converted_576floats.protobin",
#         "/mnt/lebensraum/Datasets/Day2.Equalized/Device_13/tx_10/converted_576floats.protobin",
#         "/mnt/lebensraum/Datasets/Day2.Equalized/Device_13/tx_7/converted_576floats.protobin",
#         "/mnt/lebensraum/Datasets/Day2.Equalized/Device_13/tx_4/converted_576floats.protobin",
#         "/mnt/lebensraum/Datasets/Day2.Equalized/Device_13/tx_2/converted_576floats.protobin",
#         "/mnt/lebensraum/Datasets/Day2.Equalized/Device_13/tx_5/converted_576floats.protobin",
#         "/mnt/lebensraum/Datasets/Day2.Equalized/Device_13/tx_8/converted_576floats.protobin",
#         "/mnt/lebensraum/Datasets/Day2.Equalized/Device_13/tx_9/converted_576floats.protobin",
#         "/mnt/lebensraum/Datasets/Day2.Equalized/Device_9/tx_6/converted_576floats.protobin",
#         "/mnt/lebensraum/Datasets/Day2.Equalized/Device_9/tx_3/converted_576floats.protobin",
#         "/mnt/lebensraum/Datasets/Day2.Equalized/Device_9/tx_1/converted_576floats.protobin",
#         "/mnt/lebensraum/Datasets/Day2.Equalized/Device_9/tx_10/converted_576floats.protobin",
#         "/mnt/lebensraum/Datasets/Day2.Equalized/Device_9/tx_7/converted_576floats.protobin",
#         "/mnt/lebensraum/Datasets/Day2.Equalized/Device_9/tx_4/converted_576floats.protobin",
#         "/mnt/lebensraum/Datasets/Day2.Equalized/Device_9/tx_2/converted_576floats.protobin",
#         "/mnt/lebensraum/Datasets/Day2.Equalized/Device_9/tx_5/converted_576floats.protobin",
#         "/mnt/lebensraum/Datasets/Day2.Equalized/Device_9/tx_8/converted_576floats.protobin",
#         "/mnt/lebensraum/Datasets/Day2.Equalized/Device_9/tx_9/converted_576floats.protobin",
#     ],
# }

########################################################################################################
# A ds_config for fuckin around
ds_config = {
    "size":263202,
    "cache_name": "fuckin_around",
    "train_paths":[],
    "test_paths": [
        "/mnt/lebensraum/Datasets/Day2.Equalized/Device_11/tx_6/converted_576floats.protobin",
        "/mnt/lebensraum/Datasets/Day2.Equalized/Device_11/tx_3/converted_576floats.protobin",
        "/mnt/lebensraum/Datasets/Day2.Equalized/Device_11/tx_1/converted_576floats.protobin",
        "/mnt/lebensraum/Datasets/Day2.Equalized/Device_11/tx_10/converted_576floats.protobin",
        "/mnt/lebensraum/Datasets/Day2.Equalized/Device_13/tx_2/converted_576floats.protobin",
        "/mnt/lebensraum/Datasets/Day2.Equalized/Device_13/tx_5/converted_576floats.protobin",
        "/mnt/lebensraum/Datasets/Day2.Equalized/Device_13/tx_8/converted_576floats.protobin",
        "/mnt/lebensraum/Datasets/Day2.Equalized/Device_13/tx_9/converted_576floats.protobin",
        "/mnt/lebensraum/Datasets/Day2.Equalized/Device_9/tx_6/converted_576floats.protobin",
        "/mnt/lebensraum/Datasets/Day2.Equalized/Device_9/tx_3/converted_576floats.protobin",
        "/mnt/lebensraum/Datasets/Day2.Equalized/Device_9/tx_1/converted_576floats.protobin",
        "/mnt/lebensraum/Datasets/Day2.Equalized/Device_9/tx_10/converted_576floats.protobin",
        "/mnt/lebensraum/Datasets/Day2.Equalized/Device_9/tx_7/converted_576floats.protobin",
        "/mnt/lebensraum/Datasets/Day2.Equalized/Device_9/tx_4/converted_576floats.protobin",
        "/mnt/lebensraum/Datasets/Day2.Equalized/Device_9/tx_2/converted_576floats.protobin",
        "/mnt/lebensraum/Datasets/Day2.Equalized/Device_9/tx_5/converted_576floats.protobin",
        "/mnt/lebensraum/Datasets/Day2.Equalized/Device_9/tx_8/converted_576floats.protobin",
        "/mnt/lebensraum/Datasets/Day2.Equalized/Device_9/tx_9/converted_576floats.protobin",
    ],
}


def tf_to_onehot(samples,device_id,_1,_2):
    # It works for a single y
    def to_onehot(x, y, min_y, max_y):
        z = np.zeros([max_y - min_y + 1])
        z[y-min_y] = 1

        return x,z

    return tf.py_function(
        lambda x,y: to_onehot(x,y,MIN_CLASS_LABEL, MAX_CLASS_LABEL),
        (samples,device_id),
        [tf.float32, tf.int64]
    )

def tf_filter_fn(x, device_id, transmission_id, slice_index):
    broadcast_equal = tf.math.equal(device_id, whitelist)
    return tf.math.count_nonzero(broadcast_equal) > 0  

def ds_config_to_datasets(ds_config):
    return model_utils.ds_config_to_datasets(
        ds_config, 
        lambda paths: build_dataset(paths).map(tf_to_onehot)
    )

def train_model(model, ds_config, weights_path):
    datasets = ds_config_to_datasets(ds_config)
    model_utils.train_model(model, datasets["train"], datasets["val"], weights_path, num_epochs=1)

def test_model(model, ds_config):
    datasets = ds_config_to_datasets(ds_config)
    model_utils.test_model(model, datasets["test"], MIN_CLASS_LABEL, MAX_CLASS_LABEL) 

def build_model():
    # I've been living a lie. The IQ samples have been flat this entire time. Just a stream of IQ
    # [Batches][(IQ)]

    in_shp = [2*window_size]

    model = models.Sequential()

    # Keras assumes you have channels. Well, we only have one channel, so we need to reshape the data.
    # In actuality no reshaping is happening, we are just throwing our shit into an extra dimension
    model.add(
        Reshape([1]+in_shp, input_shape=in_shp, name="classifier_1")
    )
    #<None><channel IE 1><I or Q IE 2><window_size>
    # A none value means it can be any shape
    # This is accomadating our variable batch size

    # model.add(
    #     # (symmetric_height_pad, symmetric_width_pad)
    #     # Since we index as [row][column] -> I or Q is our "height" and which sample is our "Width"
    #     # From the docs: "channels_first corresponds to inputs with shape (batch_size, channels, height, width)"
    #     ZeroPadding2D(
    #         (1, 2),
    #         data_format="channels_first",
    #         name="classifier_2"
    #     )
    # )

    model.add(
        Convolution1D(
            filters=50,
            kernel_size=7,
            strides=1,
            activation="relu",
            kernel_initializer='glorot_uniform',
            data_format="channels_first",
            name="classifier_3"
        )
    )

    # model.add(Dropout(dr))
    # model.add(ZeroPadding2D((0, 2)))

    model.add(
        Convolution1D(filters=50,
            kernel_size=7,
            strides=2,
            activation="relu",
            kernel_initializer='glorot_uniform',
            data_format="channels_first",
            name="classifier_4"
        )
    )
    model.add(Dropout(dr))
    model.add(Flatten(name="classifier_5"))

    # SM: So the number of units is the output number, input can be anything (as long as one dimensional)
    model.add(
        # Originally:     Dense(256,activation='relu',init='he_normal',name="dense1")
        Dense(
            units=256, # OG 256
            activation='relu',
            kernel_initializer='he_normal', # I ASSUME kernel is what was initialized using he_normal
            name="classifier_6"
        )
    )
    model.add(Dropout(dr))

    model.add(
        # Originally:     Dense(256,activation='relu',init='he_normal',name="dense1")
        Dense(
            units=80, # OG 256
            activation='relu',
            kernel_initializer='he_normal', # I ASSUME kernel is what was initialized using he_normal
            name="classifier_7"
        )
    )
    # model.add(Dropout(dr))

    model.add(
        # SM: Weird this did not come with an activation
        Dense(
            units=NUM_CLASSES,
            kernel_initializer='he_normal',
            name="classifier_9"
        )
    )
    model.add(Activation('softmax'))
    model.add(Reshape([NUM_CLASSES]))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model


if __name__ == "__main__":
    model = build_model()

    # <test|train> <weights path>
    if sys.argv[1] == "train":
        train_model(model, ds_config, sys.argv[2])
    if sys.argv[1] == "test":
        model.load_weights(sys.argv[2])
        test_model(model, ds_config)