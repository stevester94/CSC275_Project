#! /usr/bin/python3
import numpy as np
import tensorflow as tf
import sys
import itertools
import struct
from array import array
import gc

tf.executing_eagerly()


# From the homework, we used the following
# Input: 4+D tensor with shape: batch_shape + (channels, rows, cols)
# (1,2,N) Where N is the number of complex samples

def get_data(path, _sentinel=None, slice_num_floats=None, total_num_floats=None):
    if _sentinel != None:
        raise Exception("Use kargs")

    # Ripped from https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    def _chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    with open(path, "rb") as f:
        if total_num_floats == None:
            buf = f.read()
        else:
            buf = f.read(total_num_floats*4)

    ret_list = []
    if slice_num_floats == None:
        chunk_length=len(buf)
    else:
        chunk_length=slice_num_floats*4

        for chunk in _chunks(buf, chunk_length):
            items = int(len(chunk)/4/2)
            # The strides of an array tell us how many bytes we have to skip in memory to move to the next position along a certain axis.
            ret_list.append(np.ndarray([1,2,items], dtype=np.float32, buffer=chunk, strides=(4,4,8)))
    return ret_list

#ld = get_data(sys.argv[1], slice_num_floats=10000, total_num_floats=100000)
sliced_arrays = get_data(sys.argv[1], slice_num_floats=10000)

print("Dropping last element so we maintain uniformity")
del sliced_arrays[-1]

# Supposedly these are from the TF docs, but I ripped them from https://towardsdatascience.com/working-with-tfrecords-and-tf-train-example-36d111b3ff4d
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    # If the value is an eager tensor BytesList won't unpack a string from an EagerTensor.
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() 
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

"""
Creates a tf.train.Example message ready to be written to a file.
"""
def serialize_example(feature0, feature1, feature2, feature3):
  # Create a dictionary mapping the feature name to the tf.train.Example-compatible
  # data type.
  feature = {
      'samples': _bytes_feature(tf.io.serialize_tensor(tf.constant(feature0))),
      'device_id': _int64_feature(feature1),
      'transmission_id': _int64_feature(feature2),
      'slice_index': _int64_feature(feature3),
  }

  # Create a Features message using tf.train.Example.

  # Ugh it's a really bad naming scheme, but an example is not a single observation. In the docs they parallel arrays 10k long, being
  # stuffed into a single example. But in other examples they do it one by one. I'm assuming it just doesn't really matter.

  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return example_proto.SerializeToString()

filename = 'ramdisk/out_record'
with tf.io.TFRecordWriter(filename) as writer:
    for index,a in enumerate(sliced_arrays):
        print("Processing ", index, "/", len(sliced_arrays))
        example = serialize_example(a, 69, 420, index)
        writer.write(example)
