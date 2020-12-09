#! /usr/bin/python3
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Reshape,Dense,Dropout,Activation,Flatten, Convolution2D, MaxPooling2D, ZeroPadding2D, Permute
import tensorflow.keras.models as models
import tensorflow.keras as keras

from build_dataset import build_auto_encoder_ds, get_num_elements_in_dataset


tf.random.set_seed(1337)
DROPOUT_RATE = 0.5 # DROPOUT_RATEopout rate (%)
NUM_EPOCHS=100
BATCH_SIZE=100

TRAIN_DATASET_SIZE = 65691
TEST_DATASET_SIZE = 21648
WINDOW_SIZE = 288

# 65691 examples
train_paths = [
    ("/kek/Day_2_After_FFT/Device_9/tx_1/converted_576floats.protobin","/kek/Day_1_Equalized/Device_9/tx_1/converted_576floats.protobin"),
    ("/kek/Day_2_After_FFT/Device_9/tx_2/converted_576floats.protobin","/kek/Day_1_Equalized/Device_9/tx_2/converted_576floats.protobin"),
    ("/kek/Day_2_After_FFT/Device_9/tx_3/converted_576floats.protobin","/kek/Day_1_Equalized/Device_9/tx_3/converted_576floats.protobin"),
    ("/kek/Day_2_After_FFT/Device_9/tx_4/converted_576floats.protobin","/kek/Day_1_Equalized/Device_9/tx_4/converted_576floats.protobin"),
]

# 21648 examples
test_paths = [
    ("/kek/Day_2_After_FFT/Device_9/tx_5/converted_576floats.protobin","/kek/Day_1_Equalized/Device_9/tx_5/converted_576floats.protobin"),
]

def train_auto_encoder(model, weights_path, train_paths, test_paths):
    ds = build_auto_encoder_ds(train_paths)
    test_dataset = build_auto_encoder_ds(test_paths)

    ds = ds.take(TRAIN_DATASET_SIZE)
    ds = ds.cache(filename="auto_encoder_train_cache") # Now this is pretty fuckin cool. Delete the file to re-cache
    ds = ds.prefetch(TRAIN_DATASET_SIZE)

    test_dataset = test_dataset.cache(filename="auto_encoder_test_cache") # Now this is pretty fuckin cool. Delete the file to re-cache
    test_dataset = test_dataset.prefetch(TEST_DATASET_SIZE)


    val_size = int(0.15 * TEST_DATASET_SIZE)
    test_size = int(0.85 * TEST_DATASET_SIZE)

    train_dataset = ds.take(TRAIN_DATASET_SIZE)
    test_dataset = test_dataset.take(test_size)
    val_dataset = test_dataset.skip(test_size)
    val_dataset = test_dataset.take(val_size)

    train_dataset = train_dataset.batch(BATCH_SIZE)
    val_dataset   = val_dataset.batch(BATCH_SIZE)
    test_dataset  = test_dataset.batch(BATCH_SIZE)

    history = model.fit(
        train_dataset,
        #BATCH_SIZE=BATCH_SIZE,
        epochs=NUM_EPOCHS,
        verbose=2,
        validation_data=val_dataset,
        callbacks = [
            keras.callbacks.ModelCheckpoint(weights_path, monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
        ]
    )

def build_model():
    model = models.Sequential()

    # This is a hack. The shape of the dataset is not defined, and I can't find a way to set it, so I force it by reshaping here.
    # Defining the shape of the input is necessary... It just be like that
    in_shp=[2,WINDOW_SIZE]
    model.add(
        Reshape(in_shp, input_shape=in_shp)
    )
    model.add(Flatten())

    # SM: So the number of units is the output number, input can be anything (as long as one dimensional)
    model.add(
        # Originally:     Dense(256,activation='relu',init='he_normal',name="dense1")
        Dense(
            units=128, # OG 256
            activation='relu',
            kernel_initializer='he_normal' # I ASSUME kernel is what was initialized using he_normal
        )
    )
    model.add(
        # Originally:     Dense(256,activation='relu',init='he_normal',name="dense1")
        Dense(
            units=128, # OG 256
            activation='relu',
            kernel_initializer='he_normal' # I ASSUME kernel is what was initialized using he_normal
        )
    )
    model.add(
        # Originally:     Dense(256,activation='relu',init='he_normal',name="dense1")
        Dense(
            units=128, # OG 256
            activation='relu',
            kernel_initializer='he_normal' # I ASSUME kernel is what was initialized using he_normal
        )
    )

    model.add(
        # SM: Weird this did not come with an activation
        Dense(
            units=2*WINDOW_SIZE,
            kernel_initializer='he_normal')
    )
    # model.add(Activation('softmax'))
    model.add(Reshape([2, WINDOW_SIZE]))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model

def test_model(model, test_paths):
    test_dataset = build_auto_encoder_ds(test_paths)

    test_dataset = test_dataset.cache(filename="auto_encoder_test_cache") # Now this is pretty fuckin cool. Delete the file to re-cache
    test_dataset = test_dataset.prefetch(TEST_DATASET_SIZE)
    test_size = int(0.85 * TEST_DATASET_SIZE)
    test_dataset = test_dataset.take(test_size)
    test_dataset  = test_dataset.batch(BATCH_SIZE)

    score = model.evaluate(test_dataset, verbose=0)
    print(score)


if __name__ == "__main__":
    weights_path = 'auto_encoder.wts.h5'
    model = build_model()


    if sys.argv[1] == "train":
        train_auto_encoder(model, weights_path, train_paths, test_paths)

    if sys.argv[1] == "test":
        model.load_weights(weights_path)
        test_model(model, test_paths)
