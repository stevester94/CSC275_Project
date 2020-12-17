#! /usr/bin/python3
l = [
    "/mnt/lebensraum/Datasets/Day2.After_FFT/Device_11/tx_6/converted_576floats.protobin",
    "/mnt/lebensraum/Datasets/Day2.After_FFT/Device_11/tx_3/converted_576floats.protobin",
    "/mnt/lebensraum/Datasets/Day2.After_FFT/Device_11/tx_1/converted_576floats.protobin",
    "/mnt/lebensraum/Datasets/Day2.After_FFT/Device_11/tx_10/converted_576floats.protobin",
    "/mnt/lebensraum/Datasets/Day2.After_FFT/Device_11/tx_7/converted_576floats.protobin",
    "/mnt/lebensraum/Datasets/Day2.After_FFT/Device_11/tx_4/converted_576floats.protobin",
    "/mnt/lebensraum/Datasets/Day2.After_FFT/Device_11/tx_2/converted_576floats.protobin",
    "/mnt/lebensraum/Datasets/Day2.After_FFT/Device_11/tx_5/converted_576floats.protobin",
    "/mnt/lebensraum/Datasets/Day2.After_FFT/Device_11/tx_8/converted_576floats.protobin",
    "/mnt/lebensraum/Datasets/Day2.After_FFT/Device_11/tx_9/converted_576floats.protobin",
    "/mnt/lebensraum/Datasets/Day2.After_FFT/Device_12/tx_6/converted_576floats.protobin",
    "/mnt/lebensraum/Datasets/Day2.After_FFT/Device_12/tx_3/converted_576floats.protobin",
    "/mnt/lebensraum/Datasets/Day2.After_FFT/Device_12/tx_1/converted_576floats.protobin",
    "/mnt/lebensraum/Datasets/Day2.After_FFT/Device_12/tx_10/converted_576floats.protobin",
    "/mnt/lebensraum/Datasets/Day2.After_FFT/Device_12/tx_7/converted_576floats.protobin",
    "/mnt/lebensraum/Datasets/Day2.After_FFT/Device_12/tx_4/converted_576floats.protobin",
    "/mnt/lebensraum/Datasets/Day2.After_FFT/Device_12/tx_2/converted_576floats.protobin",
    "/mnt/lebensraum/Datasets/Day2.After_FFT/Device_12/tx_5/converted_576floats.protobin",
    "/mnt/lebensraum/Datasets/Day2.After_FFT/Device_12/tx_8/converted_576floats.protobin",
    "/mnt/lebensraum/Datasets/Day2.After_FFT/Device_12/tx_9/converted_576floats.protobin",
    "/mnt/lebensraum/Datasets/Day2.After_FFT/Device_10/tx_6/converted_576floats.protobin",
    "/mnt/lebensraum/Datasets/Day2.After_FFT/Device_10/tx_3/converted_576floats.protobin",
    "/mnt/lebensraum/Datasets/Day2.After_FFT/Device_10/tx_1/converted_576floats.protobin",
    "/mnt/lebensraum/Datasets/Day2.After_FFT/Device_10/tx_10/converted_576floats.protobin",
    "/mnt/lebensraum/Datasets/Day2.After_FFT/Device_10/tx_7/converted_576floats.protobin",
    "/mnt/lebensraum/Datasets/Day2.After_FFT/Device_10/tx_4/converted_576floats.protobin",
    "/mnt/lebensraum/Datasets/Day2.After_FFT/Device_10/tx_2/converted_576floats.protobin",
    "/mnt/lebensraum/Datasets/Day2.After_FFT/Device_10/tx_5/converted_576floats.protobin",
    "/mnt/lebensraum/Datasets/Day2.After_FFT/Device_10/tx_8/converted_576floats.protobin",
    "/mnt/lebensraum/Datasets/Day2.After_FFT/Device_10/tx_9/converted_576floats.protobin",
    "/mnt/lebensraum/Datasets/Day2.After_FFT/Device_13/tx_6/converted_576floats.protobin",
    "/mnt/lebensraum/Datasets/Day2.After_FFT/Device_13/tx_3/converted_576floats.protobin",
    "/mnt/lebensraum/Datasets/Day2.After_FFT/Device_13/tx_1/converted_576floats.protobin",
    "/mnt/lebensraum/Datasets/Day2.After_FFT/Device_13/tx_10/converted_576floats.protobin",
    "/mnt/lebensraum/Datasets/Day2.After_FFT/Device_13/tx_7/converted_576floats.protobin",
    "/mnt/lebensraum/Datasets/Day2.After_FFT/Device_13/tx_4/converted_576floats.protobin",
    "/mnt/lebensraum/Datasets/Day2.After_FFT/Device_13/tx_2/converted_576floats.protobin",
    "/mnt/lebensraum/Datasets/Day2.After_FFT/Device_13/tx_5/converted_576floats.protobin",
    "/mnt/lebensraum/Datasets/Day2.After_FFT/Device_13/tx_8/converted_576floats.protobin",
    "/mnt/lebensraum/Datasets/Day2.After_FFT/Device_13/tx_9/converted_576floats.protobin",
    "/mnt/lebensraum/Datasets/Day2.After_FFT/Device_9/tx_6/converted_576floats.protobin",
    "/mnt/lebensraum/Datasets/Day2.After_FFT/Device_9/tx_3/converted_576floats.protobin",
    "/mnt/lebensraum/Datasets/Day2.After_FFT/Device_9/tx_1/converted_576floats.protobin",
    "/mnt/lebensraum/Datasets/Day2.After_FFT/Device_9/tx_10/converted_576floats.protobin",
    "/mnt/lebensraum/Datasets/Day2.After_FFT/Device_9/tx_7/converted_576floats.protobin",
    "/mnt/lebensraum/Datasets/Day2.After_FFT/Device_9/tx_4/converted_576floats.protobin",
    "/mnt/lebensraum/Datasets/Day2.After_FFT/Device_9/tx_2/converted_576floats.protobin",
    "/mnt/lebensraum/Datasets/Day2.After_FFT/Device_9/tx_5/converted_576floats.protobin",
    "/mnt/lebensraum/Datasets/Day2.After_FFT/Device_9/tx_8/converted_576floats.protobin",
    "/mnt/lebensraum/Datasets/Day2.After_FFT/Device_9/tx_9/converted_576floats.protobin",
]

import random 
import pprint
pp = pprint.PrettyPrinter(indent=4)



random.shuffle(l)
train = l[:int(len(l)*0.7)]
test  = l[int(len(l)*0.7):]


print("train_paths =")
pp.pprint(train)

print("test_paths =")
pp.pprint(test)


assert set(train+test) == set(l)
