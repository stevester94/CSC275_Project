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

# tx_1/1589503281-0199382-ec.bin
# Ok so, we have a stream of floats in these bins, complex64 IE (float32,float32)
# We need to deinterleave these into tensors corresponding to the above shapes ^

# Yes, this works in place. Checked it with the below
#check_iters_are_equal(range(1000000000), range(1000000000))
#print(check_iters_are_equal(range(1000), range(10))) {returns false}
def check_iters_are_equal(i1, i2):
    return all(a == b for a, b in
                      itertools.zip_longest(i1, i2))

class float_from_file_iterator:
    def __init__(self, filename):
        self.cache=[]
        self.cache_size=1024
        self.file = open(filename, "rb")

    def __iter__(self):
        return self

    def __next__(self):
        if len(self.cache) == 0:
            binary = self.file.read(self.cache_size*4)
            self.cache = array("f", binary).tolist()

            # We hit the end, we're done here
            if len(self.cache) == 0:
                raise StopIteration
        return self.cache.pop()

    def __del__(self):
        self.file.close()

def deinterleave(it):
    l1 = []
    l2 = []

    while True:
        try:
            l1.append(next(it))
            l2.append(next(it))
        except:
            break

    return (l1,l2)


def even_generator(it):
    while True:
        yield next(it) 
        next(it)
     
def odd_generator(it):
    while True:
        next(it)
        yield next(it) 

def sum_even_odd(l):
    even = 0
    odd  = 0
    for i in range(len(l)):
        if i % 2 == 0:
            even += l[i]
        else:
            odd += l[i]

    print("Even: ", even, "Odd: ", odd)
        
print("Creating list")

def get_data(path):
    with open(path, "rb") as f:
        b = f.read()
        items = int(len(b)/4/2)
        # The strides of an array tell us how many bytes we have to skip in memory to move to the next position along a certain axis.
        return np.ndarray([1,2,items], dtype=np.float32, buffer=b, strides=(4,4,8))

ld = get_data("tx_1/1589503281-0199382-ec.bin")

#print(l[:20])
#print(ld[0][:10])

# The stride method is clearly fucked
#print("Even sum: ", sum(l[0::1])) # -2704.9213188713948}
#print("Odd sum: ", sum(l[1::1]))  # -2704.9634952968518}

# These are the official numbers. The iterator from file is somehow fucked
#print("Even sum: ", sum(even_generator(iter(l)))) # -1353.9999765100147
#print("Odd sum: ",  sum(odd_generator(iter(l))))  # -1350.92134236138
#sum_even_odd(l) # -1353.9999765100147 Odd:  -1350.92134236138

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

t = tf.constant(ld)



example_features = tf.train.Features(feature={
    "samples": _bytes_feature(tf.io.serialize_tensor(t)),
    "device_id": _int64_feature(69),
    "transmission_id": _int64_feature(420),
})

example_proto = tf.train.Example(features=example_features)



file_path = "muh_data"
with tf.io.TFRecordWriter(file_path) as writer:
    writer.write(example_proto)
