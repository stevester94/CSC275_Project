#! /usr/bin/python3
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Reshape,Dense,Dropout,Activation,Flatten, Convolution2D, MaxPooling2D, ZeroPadding2D
import tensorflow.keras.models as models
import tensorflow.keras as keras

from build_dataset import build_dataset,get_num_elements_in_dataset




ds = build_dataset(sys.argv[1:], 4999)

for _x,_y in ds.batch(2).take(1):
    x = _x
    y = _y

# It works for a single y
def to_onehot(y, min_y, max_y):
    z = np.zeros([max_y - min_y + 1])
    z[y-min_y] = 1

    return z


in_shp = list(x.shape[1:])
print(in_shp)

model = models.Sequential()

model.add(
    Reshape([1]+in_shp, input_shape=in_shp)
)

model.add(
    # (symmetric_height_pad, symmetric_width_pad)
    ZeroPadding2D(
        (1, 2),
        data_format="channels_first"
    )
)

print(model([[x]]))
