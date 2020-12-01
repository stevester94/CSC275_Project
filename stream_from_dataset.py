#! /usr/bin/python3
import numpy as np
import tensorflow as tf
import sys

#tf.executing_eagerly()

# From the homework, we used the following
# Input: 4+D tensor with shape: batch_shape + (channels, rows, cols)
# (1,2,N) Where N is the number of complex samples




# Now let's read this fuckin dataset
filenames = [sys.argv[1], sys.argv[2]]
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
parsed_ds = parsed_ds.prefetch(100)

# So we'll accept the single unified dataset, underneath which we have our fucktons of disparate files. we'll use pre-fetching, interleaving, all that
# special shit it make it work right, but the frontend of it will just be a regular ol dataset

# Window size is the number of IQ pairs
def sliding_window_generator(parsed_ds, window_size):
    for flat_time_domain_iq,device_id in parsed_ds:
        if window_size >= flat_time_domain_iq.shape[0]/2:
            print("Window size is >= number of IQ pairs")
            print("Window: ", window_size, "num pairs:", flat_time_domain_iq.shape[0]/2)
        window_start=0
        window_end=window_size*2

        while window_end < len(flat_time_domain_iq):
            #t = np.ones((2,window_size))
            #print(t.shape)
            #print(t.strides)
            #print(len(t.tobytes()))
            
            X_copy = flat_time_domain_iq.numpy().tobytes()
            sliced_X = X_copy[window_start*4:window_end*4]

            #print("Length of X_copy: ", len(X_copy))
            #print("Number of floats in X_copy: ", len(X_copy)/4)

            #print("Length of sliced_X: ", len(sliced_X))
            #print("number of floats in sliced_X: ", len(sliced_X)/4)

            #print("Window size: ", window_size)
            #print("window_start", window_start)
            #print("window_end", window_end)

            window = np.ndarray([2,int(window_size)], dtype=np.float32, buffer=sliced_X, strides=(4,8))

            # It looks right to me...
            #print(flat_time_domain_iq[:10])
            #print(window[0][:5])
            #print(window[1][:5])
         
            yield window, device_id 

            window_start += 1
            window_end   += 1

    print("Raise")
    raise StopIteration

# 35245 examples in here
# 0m10.458s
#regular_count = 0
#for f1,f2 in parsed_ds:
    #regular_count += 1
    #print(regular_count)

# window size of 4999: real    0m12.196s
windowed_count = 0
set_of_dev_ids = set()
for f1,f2 in sliding_window_generator(parsed_ds, 4999):
    windowed_count += 1
    set_of_dev_ids.add(f2.numpy())
print(windowed_count)
print(set_of_dev_ids)
