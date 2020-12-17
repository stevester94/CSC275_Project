#! /usr/bin/python3
import numpy as np
import tensorflow as tf
from build_dataset import build_dataset

def get_data(path, slice_num_floats):
    # Ripped from https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    def _chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    with open(path, "rb") as f:
        buf = f.read()

    chunk_length=slice_num_floats*4

    for chunk in _chunks(buf, chunk_length):
        items = int(len(chunk)/4)
        yield np.ndarray([items], dtype=np.float32, buffer=chunk)


min_class_label=9
max_class_label=13

def tf_to_onehot(samples,device_id,_1,_2):
    # It works for a single y
    def to_onehot(x, y, min_y, max_y):
        z = np.zeros([max_y - min_y + 1])
        z[y-min_y] = 1

        return x,z

    return tf.py_function(
        lambda x,y: to_onehot(x,y,min_class_label, max_class_label),
        (samples,device_id),
        [tf.float32, tf.int64]
    )


train_paths = [
    '/mnt/lebensraum/Datasets/Day1.Equalized/Device_9/tx_7/converted_576floats.protobin'
]

test_dataset = build_dataset(train_paths)
test_dataset = test_dataset.map(tf_to_onehot)

it = get_data("/mnt/lebensraum/Datasets/Day1.Equalized/Device_9/tx_7/1587959509-4155354-fb.bin", 576)

for x in test_dataset:
    # assert np.array_equal(x[0], next(it))
    print(x[0])

    sys.exit(1)