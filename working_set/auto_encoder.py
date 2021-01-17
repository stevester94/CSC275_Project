#! /usr/bin/python3
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Reshape,Dense,Dropout,Activation,Flatten, Convolution2D, MaxPooling2D, ZeroPadding2D, Permute
import tensorflow.keras.models as models
import tensorflow.keras as keras

from build_dataset import build_auto_encoder_ds, get_num_elements_in_dataset
import model_utils

tf.random.set_seed(1337)

DROPOUT_RATE = 0.5 # DROPOUT_RATEopout rate (%)
WINDOW_SIZE = 288

#######################################################################################################
# This is the goal dataset
# workspace: encoder.shuffled
ds_config = {
    "size": 191907,
    "cache_name": "encoder_target",
    "paths": [   
        (   '/mnt/lebensraum/Datasets/Day2.After_FFT/Device_9/tx_10/converted_576floats.protobin',
            '/mnt/lebensraum/Datasets/Day1.Equalized/Device_9/tx_10/converted_576floats.protobin'),
        (   '/mnt/lebensraum/Datasets/Day2.After_FFT/Device_9/tx_3/converted_576floats.protobin',
            '/mnt/lebensraum/Datasets/Day1.Equalized/Device_9/tx_3/converted_576floats.protobin'),
        (   '/mnt/lebensraum/Datasets/Day2.After_FFT/Device_9/tx_8/converted_576floats.protobin',
            '/mnt/lebensraum/Datasets/Day1.Equalized/Device_9/tx_8/converted_576floats.protobin'),
        (   '/mnt/lebensraum/Datasets/Day2.After_FFT/Device_9/tx_6/converted_576floats.protobin',
            '/mnt/lebensraum/Datasets/Day1.Equalized/Device_9/tx_6/converted_576floats.protobin'),
        (   '/mnt/lebensraum/Datasets/Day2.After_FFT/Device_9/tx_4/converted_576floats.protobin',
            '/mnt/lebensraum/Datasets/Day1.Equalized/Device_9/tx_4/converted_576floats.protobin'),
        (   '/mnt/lebensraum/Datasets/Day2.After_FFT/Device_9/tx_7/converted_576floats.protobin',
            '/mnt/lebensraum/Datasets/Day1.Equalized/Device_9/tx_7/converted_576floats.protobin'),
        (   '/mnt/lebensraum/Datasets/Day2.After_FFT/Device_9/tx_2/converted_576floats.protobin',
            '/mnt/lebensraum/Datasets/Day1.Equalized/Device_9/tx_2/converted_576floats.protobin'),
        (   '/mnt/lebensraum/Datasets/Day2.After_FFT/Device_9/tx_5/converted_576floats.protobin',
            '/mnt/lebensraum/Datasets/Day1.Equalized/Device_9/tx_5/converted_576floats.protobin'),
        (   '/mnt/lebensraum/Datasets/Day2.After_FFT/Device_9/tx_1/converted_576floats.protobin',
            '/mnt/lebensraum/Datasets/Day1.Equalized/Device_9/tx_1/converted_576floats.protobin'),
        (   '/mnt/lebensraum/Datasets/Day2.After_FFT/Device_9/tx_9/converted_576floats.protobin',
            '/mnt/lebensraum/Datasets/Day1.Equalized/Device_9/tx_9/converted_576floats.protobin'),
    ]
}

#######################################################################################################
# This is the dummy auto encoder set
# going to just bump up the neural net so we can so a strict passthrough, verifying the architecture
# After 14 epochs get 0 loss
# DATASET_SIZE=191907
# train_paths = [   
#     (   '/mnt/lebensraum/Datasets/Day1.Equalized/Device_9/tx_10/converted_576floats.protobin',
#         '/mnt/lebensraum/Datasets/Day1.Equalized/Device_9/tx_10/converted_576floats.protobin'),
#     (   '/mnt/lebensraum/Datasets/Day1.Equalized/Device_9/tx_3/converted_576floats.protobin',
#         '/mnt/lebensraum/Datasets/Day1.Equalized/Device_9/tx_3/converted_576floats.protobin'),
#     (   '/mnt/lebensraum/Datasets/Day1.Equalized/Device_9/tx_8/converted_576floats.protobin',
#         '/mnt/lebensraum/Datasets/Day1.Equalized/Device_9/tx_8/converted_576floats.protobin'),
#     (   '/mnt/lebensraum/Datasets/Day1.Equalized/Device_9/tx_6/converted_576floats.protobin',
#         '/mnt/lebensraum/Datasets/Day1.Equalized/Device_9/tx_6/converted_576floats.protobin'),
#     (   '/mnt/lebensraum/Datasets/Day1.Equalized/Device_9/tx_4/converted_576floats.protobin',
#         '/mnt/lebensraum/Datasets/Day1.Equalized/Device_9/tx_4/converted_576floats.protobin'),
#     (   '/mnt/lebensraum/Datasets/Day1.Equalized/Device_9/tx_7/converted_576floats.protobin',
#         '/mnt/lebensraum/Datasets/Day1.Equalized/Device_9/tx_7/converted_576floats.protobin'),
#     (   '/mnt/lebensraum/Datasets/Day1.Equalized/Device_9/tx_2/converted_576floats.protobin',
#         '/mnt/lebensraum/Datasets/Day1.Equalized/Device_9/tx_2/converted_576floats.protobin'),
#     (   '/mnt/lebensraum/Datasets/Day1.Equalized/Device_9/tx_5/converted_576floats.protobin',
#         '/mnt/lebensraum/Datasets/Day1.Equalized/Device_9/tx_5/converted_576floats.protobin'),
#     (   '/mnt/lebensraum/Datasets/Day1.Equalized/Device_9/tx_1/converted_576floats.protobin',
#         '/mnt/lebensraum/Datasets/Day1.Equalized/Device_9/tx_1/converted_576floats.protobin'),
#     (   '/mnt/lebensraum/Datasets/Day1.Equalized/Device_9/tx_9/converted_576floats.protobin',
#         '/mnt/lebensraum/Datasets/Day1.Equalized/Device_9/tx_9/converted_576floats.protobin'),
# ]
# test_paths = []

def ds_config_to_datasets(ds_config):
    return model_utils.ds_config_to_datasets(
        ds_config, 
        lambda paths: build_auto_encoder_ds(paths)
    )

def build_cache(ds_config):
    datasets = ds_config_to_datasets(ds_config)

    print("Populating training set")
    get_num_elements_in_dataset(datasets["train"])

    print("Populating test set")
    get_num_elements_in_dataset(datasets["test"])
    
    print("Populating val set")
    get_num_elements_in_dataset(datasets["val"])

def train_model(model, ds_config, weights_path):
    datasets = ds_config_to_datasets(ds_config)
    return model_utils.train_model(model, datasets["train"], datasets["val"], weights_path, num_epochs=100)

def test_model(model, ds_config):
    datasets = ds_config_to_datasets(ds_config)
    model_utils.test_model(model, datasets["test"], MIN_CLASS_LABEL, MAX_CLASS_LABEL, "muh_confusion.png") 

def build_model():
    model = models.Sequential()

    # This is a hack. The shape of the dataset is not defined, and I can't find a way to set it, so I force it by reshaping here.
    # Defining the shape of the input is necessary... It just be like that
    in_shp=[2*WINDOW_SIZE]
    model.add(
        Reshape(in_shp, input_shape=in_shp)
    )

    NUM_DENSE_NODES=1000 # Originally 128
    # SM: So the number of units is the output number, input can be anything (as long as one dimensional)
    model.add(Dense(units=NUM_DENSE_NODES,activation='relu',kernel_initializer='he_normal',name="auto_encoder_1"))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(units=NUM_DENSE_NODES,activation='relu',kernel_initializer='he_normal',name="auto_encoder_2"))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(units=NUM_DENSE_NODES,activation='relu',kernel_initializer='he_normal',name="auto_encoder_3"))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(units=NUM_DENSE_NODES,activation='relu',kernel_initializer='he_normal',name="auto_encoder_4"))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(units=NUM_DENSE_NODES,activation='relu',kernel_initializer='he_normal',name="auto_encoder_5"))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(units=NUM_DENSE_NODES,activation='relu',kernel_initializer='he_normal',name="auto_encoder_6"))
    model.add(Dropout(DROPOUT_RATE))


    model.add(
        # SM: Weird this did not come with an activation
        Dense(
            units=2*WINDOW_SIZE,
            kernel_initializer='he_normal',
            name="auto_encoder_out"
        )
    )
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model



if __name__ == "__main__":
    model = build_model()
    workspace_path = sys.argv[1]

    if sys.argv[1] == "train":
        history = train_model(model, ds_config, sys.argv[2])
        model_utils.plot_loss_curve(history, "muh_loss.png")
    if sys.argv[1] == "test":
        pass