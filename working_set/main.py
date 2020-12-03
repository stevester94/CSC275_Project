#! /usr/bin/python3
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Reshape,Dense,Dropout,Activation,Flatten, Convolution2D, MaxPooling2D, ZeroPadding2D
import tensorflow.keras.models as models
import tensorflow.keras as keras

from build_dataset import build_dataset,get_num_elements_in_dataset




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


result = model(x)

print("Result shape: ", result.shape)
