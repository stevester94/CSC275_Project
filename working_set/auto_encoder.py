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

WINDOW_SIZE = 288
prefetch_size=1000

TRAIN_RATIO=0.7
VALIDATION_RATIO=0.2
TEST_RATIO=0.1

batch_size=250
dr = 0.5 # dropout rate (%)


WEIGHTS_NAME="encoder.wts.h5"
TEST_CACHE_NAME="encoder_test_cache"
TRAIN_CACHE_NAME="encoder_train_cache"
VAL_CACHE_NAME="encoder_val_cache"

DATASET_SIZE=191907
train_paths = [   
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
test_paths = []


def train_model(model, workspace_path, train_paths, test_paths):
    dataset = build_auto_encoder_ds(train_paths+test_paths)    
    dataset = dataset.cache(filename=workspace_path+"/"+"shuffled_dataset")
    
    # DATASET_SIZE = get_num_elements_in_dataset(dataset)
    print("Dataset size is: ", DATASET_SIZE)
    
    dataset = dataset.shuffle(DATASET_SIZE, seed=1337, reshuffle_each_iteration=False)
    

    train_dataset_size = int(DATASET_SIZE * TRAIN_RATIO)
    val_dataset_size   = int(DATASET_SIZE * VALIDATION_RATIO)
    test_dataset_size  = int(DATASET_SIZE * TEST_RATIO)

    train_dataset = dataset.take(train_dataset_size)
    train_dataset = train_dataset.cache(filename=workspace_path+"/"+TRAIN_CACHE_NAME) # Now this is pretty fuckin cool. Delete the file to re-cache
    train_dataset = train_dataset.prefetch(prefetch_size)

    val_dataset = dataset.skip(train_dataset_size).take(val_dataset_size)
    val_dataset = val_dataset.cache(filename=workspace_path+"/"+VAL_CACHE_NAME) # Now this is pretty fuckin cool. Delete the file to re-cache
    val_dataset = val_dataset.prefetch(prefetch_size)

    test_dataset = dataset.skip(train_dataset_size+val_dataset_size).take(test_dataset_size)
    test_dataset = test_dataset.cache(filename=workspace_path+"/"+TEST_CACHE_NAME) # Now this is pretty fuckin cool. Delete the file to re-cache
    test_dataset = test_dataset.prefetch(prefetch_size)

    # Build our batches
    train_dataset = train_dataset.batch(batch_size)
    val_dataset   = val_dataset.batch(batch_size)

    history = model.fit(
        train_dataset,
        #BATCH_SIZE=BATCH_SIZE,
        epochs=NUM_EPOCHS,
        verbose=2,
        validation_data=val_dataset,
        callbacks = [
            keras.callbacks.ModelCheckpoint(workspace_path+"/"+WEIGHTS_NAME, monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
        ]
    )

def build_model():
    model = models.Sequential()

    # This is a hack. The shape of the dataset is not defined, and I can't find a way to set it, so I force it by reshaping here.
    # Defining the shape of the input is necessary... It just be like that
    in_shp=[2*WINDOW_SIZE]
    model.add(
        Reshape(in_shp, input_shape=in_shp)
    )

    # SM: So the number of units is the output number, input can be anything (as long as one dimensional)
    model.add(
        # Originally:     Dense(256,activation='relu',init='he_normal',name="dense1")
        Dense(
            units=128, # OG 256
            activation='relu',
            kernel_initializer='he_normal', # I ASSUME kernel is what was initialized using he_normal
            name="auto_encoder_1"
        )
    )
    model.add(Dropout(dr))
    model.add(
        # Originally:     Dense(256,activation='relu',init='he_normal',name="dense1")
        Dense(
            units=128, # OG 256
            activation='relu',
            kernel_initializer='he_normal', # I ASSUME kernel is what was initialized using he_normal
            name="auto_encoder_2"
        )
    )
    model.add(Dropout(dr))
    model.add(
        # Originally:     Dense(256,activation='relu',init='he_normal',name="dense1")
        Dense(
            units=128, # OG 256
            activation='relu',
            kernel_initializer='he_normal', # I ASSUME kernel is what was initialized using he_normal
            name="auto_encoder_3"
        )
    )
    model.add(Dropout(dr))
    model.add(
        # SM: Weird this did not come with an activation
        Dense(
            units=2*WINDOW_SIZE,
            kernel_initializer='he_normal',
            name="auto_encoder_4"
        )
    )
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model

def test_model(model, workspace_path, test_paths):
    test_dataset = build_auto_encoder_ds(test_paths)   
    test_dataset = test_dataset.cache(filename="auto_encoder_test_cache") # Now this is pretty fuckin cool. Delete the file to re-cache
    test_dataset = test_dataset.prefetch(prefetch_size)


    score = model.evaluate(test_dataset, verbose=0)
    print(score)


if __name__ == "__main__":
    model = build_model()
    workspace_path = sys.argv[1]

    if sys.argv[2] == "train":
        
        train_model(model, workspace_path, train_paths, test_paths)
    if sys.argv[2] == "test":
        # <workspace path> <test> <weights path>
        weights_path = sys.argv[3]
        model.load_weights(weights_path)
        test_model(model, workspace_path, test_paths)