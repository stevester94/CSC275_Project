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

with open("tx_1/ramdisk/1589503281-0199382-ec.bin", "rb") as f:
    b = f.read()
    items = int(len(b)/4/2)
    # The strides of an array tell us how many bytes we have to skip in memory to move to the next position along a certain axis.
    ld = np.ndarray([2,items], dtype=np.float32, buffer=b, strides=(4, 8))

with open("tx_1/ramdisk/1589503281-0199382-ec.bin", "rb") as f:
    l = np.fromfile(f, dtype=np.float32, count=-1)

print(l[:20])
print(ld[0][:10])

# The stride method is clearly fucked
#print("Even sum: ", sum(l[0::1])) # -2704.9213188713948}
#print("Odd sum: ", sum(l[1::1]))  # -2704.9634952968518}

# The ndarray is working!
#print("0 sum: ", sum(ld[0])) # -1353.9999765100147
#print("1 sum: ", sum(ld[1])) # -1350.92134236138

# These are the official numbers. The iterator from file is somehow fucked
#print("Even sum: ", sum(even_generator(iter(l)))) # -1353.9999765100147
#print("Odd sum: ",  sum(odd_generator(iter(l))))  # -1350.92134236138
#sum_even_odd(l) # -1353.9999765100147 Odd:  -1350.92134236138

#print("Even sum: ", sum(even_generator(float_from_file_iterator("tx_1/ramdisk/1589503281-0199382-ec.bin")))) # -1350.92134236138
#print("Odd sum: ",  sum(odd_generator(float_from_file_iterator("tx_1/ramdisk/1589503281-0199382-ec.bin"))))  # -1353.9999765100147
#sum_even_odd(l) # Even:  -1353.9999765100147 Odd:  -1350.92134236138


#print("iter")
#iter_sum = sum(numpy_from_file_iterator("tx_1/1589503281-0199382-ec.bin")) #-2704.9213188713948, 1m39.148s
#print("conventional")
#conventional_sum =  sum(np.fromfile("tx_1/1589503281-0199382-ec.bin", np.float32, -1)) #-2704.9213188713948, 1m24.416s

#even_gen = even_generator(float_from_file_iterator("tx_1/ramdisk/1589503281-0199382-ec.bin"))
#odd_gen  = odd_generator(float_from_file_iterator("tx_1/ramdisk/1589503281-0199382-ec.bin"))

#even,odd = deinterleave(float_from_file_iterator("tx_1/ramdisk/1589503281-0199382-ec.bin"))
#l = list(float_from_file_iterator("tx_1/ramdisk/1589503281-0199382-ec.bin"))

#print("Creating Constant")
#t_even = tf.constant(l_even)
#t_odd = tf.constant(l_odd)

#t = tf.constant(
    #[
        #l_2d
    #]
#)

#l_even = None
#l_odd  = None
#gc.collect()


#[
    #[list(even_gen),list(odd_gen)]
#])
    
#[[2 3]
#[4 5]], shape=(1,2, 2), dtype=float32)    
