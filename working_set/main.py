#! /usr/bin/python3
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Reshape,Dense,Dropout,Activation,Flatten, Convolution2D, MaxPooling2D, ZeroPadding2D, Permute
import tensorflow.keras.models as models
import tensorflow.keras as keras

from build_dataset import build_dataset,get_num_elements_in_dataset

min_class_label=0
max_class_label=10
num_classes=max_class_label-min_class_label+1

ds = build_dataset(sys.argv[1:], 4999)

for _x,_y in ds.batch(3).take(1):
    x = _x
    y = _y

# It works for a single y
def to_onehot(y, min_y, max_y):
    z = np.zeros([max_y - min_y + 1])
    z[y-min_y] = 1

    return z

# [Batches][Channel (I or Q)][samples]
# <batches><2><window size>
print("Top level input shape: ", list(x.shape))

in_shp = list(x.shape[1:])
print("Functional input shape: ", in_shp)
dr = 0.5 # dropout rate (%)

model = models.Sequential()

# Keras assumes you have channels. Well, we only have one channel, so we need to reshape the data.
# In actuality no reshaping is happening, we are just throwing our shit into an extra dimension
model.add(
    Reshape([1]+in_shp, input_shape=in_shp)
)
#<None><channel IE 1><I or Q IE 2><window_size>
# A none value means it can be any shape
# This is accomadating our variable batch size
print("Shape after reshape layer: ", model.output_shape)



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
