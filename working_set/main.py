#! /usr/bin/python3
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Reshape,Dense,Dropout,Activation,Flatten, Convolution2D, MaxPooling2D, ZeroPadding2D, Permute
import tensorflow.keras.models as models
import tensorflow.keras as keras

from build_dataset import build_dataset,get_num_elements_in_dataset

min_class_label=9
max_class_label=10
num_classes=max_class_label-min_class_label+1

dr = 0.5 # dropout rate (%)
nb_epoch=100
batch_size=100

window_size=288
DATASET_SIZE = 100000

paths = [
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_11/tx_6/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_11/tx_3/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_11/tx_1/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_11/tx_10/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_11/tx_7/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_11/tx_4/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_11/tx_2/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_11/tx_5/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_11/tx_8/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_11/tx_9/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_12/tx_6/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_12/tx_3/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_12/tx_1/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_12/tx_10/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_12/tx_7/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_12/tx_4/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_12/tx_2/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_12/tx_5/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_12/tx_8/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_12/tx_9/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_10/tx_6/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_10/tx_3/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_10/tx_1/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_10/tx_10/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_10/tx_7/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_10/tx_4/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_10/tx_2/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_10/tx_5/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_10/tx_8/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_10/tx_9/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_13/tx_6/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_13/tx_3/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_13/tx_1/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_13/tx_10/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_13/tx_7/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_13/tx_4/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_13/tx_2/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_13/tx_5/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_13/tx_8/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_13/tx_9/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_9/tx_6/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_9/tx_3/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_9/tx_1/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_9/tx_10/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_9/tx_7/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_9/tx_4/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_9/tx_2/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_9/tx_5/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_9/tx_8/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_9/tx_9/converted_576floats.protobin",
]


# It works for a single y
def to_onehot(x, y, min_y, max_y):
    z = np.zeros([max_y - min_y + 1])
    z[y-min_y] = 1

    return x,z

def tf_to_onehot(x,y):
    return tf.py_function(
        lambda x,y: to_onehot(x,y,min_class_label, max_class_label),
        (x,y),
        [tf.float32, tf.int64]
    )

ds = build_dataset(paths)

whitelist = tf.constant([9, 10], dtype=tf.int64)
def tf_filter_fn(_, y):
    broadcast_equal = tf.math.equal(y, whitelist)
    return tf.math.count_nonzero(broadcast_equal) > 0

ds = ds.filter(tf_filter_fn) # We only want device 9 or 10
ds = ds.take(DATASET_SIZE)
ds = ds.map(tf_to_onehot)
ds = ds.cache(filename="muh_cache") # Now this is pretty fuckin cool. Delete the file to re-cache
ds = ds.prefetch(DATASET_SIZE)

train_size = int(0.7 * DATASET_SIZE)
val_size = int(0.15 * DATASET_SIZE)
test_size = int(0.15 * DATASET_SIZE)

train_dataset = ds.take(train_size)
test_dataset = ds.skip(train_size)
val_dataset = test_dataset.skip(val_size)
test_dataset = test_dataset.take(test_size)

train_dataset = train_dataset.batch(batch_size)
val_dataset   = val_dataset.batch(batch_size)
test_dataset  = test_dataset.batch(batch_size)


# [Batches][Channel (I or Q)][samples]
# <batches><2><window size>

#in_shp = list(x.shape[1:])
in_shp = [2,window_size]

model = models.Sequential()

# Keras assumes you have channels. Well, we only have one channel, so we need to reshape the data.
# In actuality no reshaping is happening, we are just throwing our shit into an extra dimension
model.add(
    Reshape([1]+in_shp, input_shape=in_shp)
)
#<None><channel IE 1><I or Q IE 2><window_size>
# A none value means it can be any shape
# This is accomadating our variable batch size
print("Reshape input shape: ", model.input_shape)
print("Reshape output shape: ", model.output_shape)

#print(in_shp)
#print(model(x))
#sys.exit(1)


model.add(
    # (symmetric_height_pad, symmetric_width_pad)
    # Since we index as [row][column] -> I or Q is our "height" and which sample is our "Width"
    # From the docs: "channels_first corresponds to inputs with shape (batch_size, channels, height, width)"
    ZeroPadding2D(
        (1, 2),
        data_format="channels_first"
    )
)
print("Shape after padding: ", model.output_shape)

# Errors out with 
# "The Conv2D op currently only supports the NHWC tensor format on the CPU. The op was given the format: NCHW [Op:Conv2D]"
# This is saying CPU doesn't like channels first
# Something like this: https://stackoverflow.com/questions/37689423/convert-between-nhwc-and-nchw-in-tensorflow
# out = tf.transpose(x, [0, 2, 3, 1])
model.add(
# Originally: Convolution2D(256,1,3, border_mode='valid', activation="relu", name="conv1", init='glorot_uniform')
    Convolution2D(
        filters=256,
        kernel_size=3,
        strides=1,
        activation="relu",
        kernel_initializer='glorot_uniform',
        data_format="channels_first"
    )
)

model.add(Dropout(dr))
model.add(ZeroPadding2D((0, 2)))

print("Convolution2D (Second) Input Shape", model.output_shape)
model.add(
    Convolution2D(filters=80,
        kernel_size=3, # SM WARNING: Originally 2
        strides=2,
        activation="relu",
        kernel_initializer='glorot_uniform',
        data_format="channels_first"
    )
)

model.add(Dropout(dr))



model.add(Flatten())
print("Flatten Output Shape: ", model.output_shape)

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

print("Dense (First) Output Shape: ", model.output_shape)

model.add(Dropout(dr))
model.add(
    # SM: Weird this did not come with an activation
    Dense(
        units=num_classes,
        kernel_initializer='he_normal')
)
model.add(Activation('softmax'))
model.add(Reshape([num_classes]))
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.summary()


###########
# Block 7 #
###########

# perform training ...
#   - call the main training loop in keras for our network+dataset

filepath = 'steve.wts.h5'

history = model.fit(
    train_dataset,
    #batch_size=batch_size,
    epochs=nb_epoch,
    verbose=2,
    validation_data=val_dataset,
    callbacks = [
        keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
    ]
)
