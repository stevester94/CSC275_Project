#! /usr/bin/python3
import numpy as np
import tensorflow as tf
import sys

#tf.executing_eagerly()

# From the homework, we used the following
# Input: 4+D tensor with shape: batch_shape + (channels, rows, cols)
# (1,2,N) Where N is the number of complex samples



filename = sys.argv[1]

# Now let's read this fuckin dataset
filenames = [filename]
raw_dataset = tf.data.TFRecordDataset(filenames)

def deserialize_example(raw_record):
  global k
  example = tf.train.Example()
  example.ParseFromString(raw_record.numpy())

  tens = tf.io.parse_tensor(example.features.feature["samples"].bytes_list.value[0], np.float32)

  return [tens, example.features.feature["device_id"].int64_list.value[0]]
  #return [tens, example.features.feature["device_id"].int64_list.value[0], example.features.feature["transmission_id"].int64_list.value[0]]

"""
Wrapper to use our deserialize_example in TF
"""
def tf_deserialize_example(raw_record):
  return tf.py_function(
    deserialize_example,
    (raw_record,),  # pass these args to the above function.
    [tf.float32, tf.int64])      # [samples], device_id


parsed_ds = raw_dataset.map(tf_deserialize_example)

for f1,f2 in parsed_ds:
    print(f1)
    print(f2)
