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
parsed_ds = parsed_ds.prefetch(10)

# So we'll accept the single unified dataset, underneath which we have our fucktons of disparate files. we'll use pre-fetching, interleaving, all that
# special shit it make it work right, but the frontend of it will just be a regular ol dataset

# This is incredibly slow
def sliding_window_generator(parsed_ds, window_size):
    #print("Generator entrypint")
    for X,Y in parsed_ds:
        #print("for loop")

        window_start=0
        window_end=window_size

        X_i = X[0][0]
        X_q = X[0][1]

        while len(X_i[window_start:window_end]) == window_size:
            #print("while loop")
            
            ar = np.array([   [X_i[window_start:window_end], X_q[window_start:window_end]]   ])
            
            yield ar, Y

            window_start += 1
            window_end   += 1
    

    print("Raise")
    raise StopIteration
            
regular_count = 0
for f1,f2 in parsed_ds:
    regular_count += 1
print(regular_count)

windowed_count = 0
for f1,f2 in sliding_window_generator(parsed_ds, 1000):
    windowed_count += 1
    print(windowed_count)
