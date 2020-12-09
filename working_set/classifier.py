#! /usr/bin/python3
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Reshape,Dense,Dropout,Activation,Flatten, Convolution2D, MaxPooling2D, ZeroPadding2D, Permute
import tensorflow.keras.models as models
import tensorflow.keras as keras

import matplotlib.pyplot as plt

from build_dataset import build_dataset,get_num_elements_in_dataset, build_debug_set

tf.random.set_seed(1337)
min_class_label=9
#max_class_label=10
max_class_label=13
num_classes=max_class_label-min_class_label+1

dr = 0.5 # dropout rate (%)
nb_epoch=100
batch_size=100

window_size=288
DATASET_SIZE = 100000
#DATASET_SIZE = 1000000
# Between two sets we have 1053403

paths = [
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_11/tx_6/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_11/tx_3/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_11/tx_1/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_11/tx_10/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_11/tx_7/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_11/tx_4/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_11/tx_2/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_11/tx_5/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_11/tx_8/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_11/tx_9/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_12/tx_6/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_12/tx_3/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_12/tx_1/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_12/tx_10/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_12/tx_7/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_12/tx_4/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_12/tx_2/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_12/tx_5/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_12/tx_8/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_12/tx_9/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_10/tx_6/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_10/tx_3/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_10/tx_1/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_10/tx_10/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_10/tx_7/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_10/tx_4/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_10/tx_2/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_10/tx_5/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_10/tx_8/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_10/tx_9/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_13/tx_6/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_13/tx_3/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_13/tx_1/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_13/tx_10/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_13/tx_7/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_13/tx_4/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_13/tx_2/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_13/tx_5/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_13/tx_8/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_13/tx_9/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_9/tx_6/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_9/tx_3/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_9/tx_1/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_9/tx_10/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_9/tx_7/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_9/tx_4/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_9/tx_2/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_9/tx_5/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_9/tx_8/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_9/tx_9/converted_576floats.protobin",
]

# 1053403 examples
train_paths = [
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_9/tx_1/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_10/tx_1/converted_576floats.protobin",
]



# 736348 examples
test_paths = [
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_9/tx_2/converted_576floats.protobin",
    "/kek/Day_1_Before_FFT/Devices_1_through_5/Device_10/tx_2/converted_576floats.protobin",
]


##################
# Equalized sets #
##################
train_paths = [
    "/kek/Day_1_Equalized/Device_9/tx_1/converted_576floats.protobin",
    "/kek/Day_1_Equalized/Device_10/tx_1/converted_576floats.protobin",
]

test_paths = [
    "/kek/Day_1_Equalized/Device_9/tx_2/converted_576floats.protobin",
    "/kek/Day_1_Equalized/Device_10/tx_2/converted_576floats.protobin",
]

whitelist = tf.constant([9, 10], dtype=tf.int64)

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, labels=[]):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



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

def tf_filter_fn(x, device_id, transmission_id, slice_index):
    broadcast_equal = tf.math.equal(device_id, whitelist)
    return tf.math.count_nonzero(broadcast_equal) > 0

def train_model(model, weights_path, train_paths, test_paths):
    ds = build_dataset(train_paths)
    test_dataset = build_dataset(test_paths)

    #ds = ds.filter(tf_filter_fn) # We only want device 9 or 10
    ds = ds.take(DATASET_SIZE)
    ds = ds.map(tf_to_onehot)
    ds = ds.cache(filename="classifier_train_cache") # Now this is pretty fuckin cool. Delete the file to re-cache
    ds = ds.prefetch(DATASET_SIZE)

    test_dataset = test_dataset.map(tf_to_onehot)
    test_dataset = test_dataset.cache(filename="classifier_test_cache") # Now this is pretty fuckin cool. Delete the file to re-cache
    test_dataset = test_dataset.prefetch(DATASET_SIZE)

    # train_size = int(0.7 * DATASET_SIZE)
    val_size = int(0.15 * DATASET_SIZE)
    test_size = int(0.15 * DATASET_SIZE)

    train_dataset = ds.take(DATASET_SIZE)
    test_dataset = test_dataset.take(test_size)
    val_dataset = test_dataset.skip(test_size)
    val_dataset = test_dataset.take(val_size)

    train_dataset = train_dataset.batch(batch_size)
    val_dataset   = val_dataset.batch(batch_size)
    test_dataset  = test_dataset.batch(batch_size)

    history = model.fit(
        train_dataset,
        #batch_size=batch_size,
        epochs=nb_epoch,
        verbose=2,
        validation_data=val_dataset,
        callbacks = [
            keras.callbacks.ModelCheckpoint(weights_path, monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
        ]
    )

    

def test_model(model, test_paths):
    test_dataset = build_dataset(test_paths)
    test_dataset = test_dataset.map(tf_to_onehot)
    test_dataset = test_dataset.cache(filename="classifier_test_cache") # Now this is pretty fuckin cool. Delete the file to re-cache
    test_dataset = test_dataset.prefetch(DATASET_SIZE)
    test_size = int(0.15 * DATASET_SIZE)
    test_dataset = test_dataset.take(test_size)
    test_dataset  = test_dataset.batch(batch_size)

    test_Y_hat = model.predict(test_dataset)
    conf = np.zeros([num_classes,num_classes])
    confnorm = np.zeros([num_classes,num_classes])

    counter = 0
    correct = 0
    incorrect = 0
    for x,y in test_dataset.unbatch(): # Batching was making this weird
        
        actual    = int(np.argmax(y))
        predicted = int(np.argmax(test_Y_hat[counter,:]))

        if actual == predicted:
            correct += 1
        else:
            incorrect += 1

        conf[actual, predicted] = conf[actual, predicted] + 1 

        counter += 1

    for i in range(0,num_classes):
        confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])

    print("Correct: ", correct)
    print("Incorrect: ", incorrect)
    print("Accuracy: ", correct / (correct + incorrect))
    print(conf)
    print(confnorm)

    # plot_confusion_matrix(confnorm, labels=list(range(min_class_label, max_class_label+1)))
    # plt.savefig("confusion.png")
    # #plt.figure()

def build_model():
    # [Batches][Channel (I or Q)][samples]
    # <batches><2><window size>

    in_shp = [2,window_size]

    model = models.Sequential()

    # Keras assumes you have channels. Well, we only have one channel, so we need to reshape the data.
    # In actuality no reshaping is happening, we are just throwing our shit into an extra dimension
    model.add(
        Reshape([1]+in_shp, input_shape=in_shp)
    )
    #<None><channel IE 1><I or Q IE 2><window_size>
    # A none value means it can be any shape
    # This is accomadating our variable batch size

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

    model.add(Dropout(dr))
    model.add(ZeroPadding2D((0, 2)))

    print("Convolution2D (Second) Input Shape", model.output_shape)
    model.add(
        Convolution2D(filters=80,
            kernel_size=3, # SM WARNING: Originally 2
            strides=2,
            activation="relu",
            kernel_initializer='glorot_uniform',
            data_format="channels_first"
        )
    )

    model.add(Dropout(dr))


    model.add(Flatten())

    # SM: So the number of units is the output number, input can be anything (as long as one dimensional)
    model.add(
        # Originally:     Dense(256,activation='relu',init='he_normal',name="dense1")
        Dense(
            units=128, # OG 256
            activation='relu',
            kernel_initializer='he_normal' # I ASSUME kernel is what was initialized using he_normal
        )
    )
    model.add(
        # Originally:     Dense(256,activation='relu',init='he_normal',name="dense1")
        Dense(
            units=128, # OG 256
            activation='relu',
            kernel_initializer='he_normal' # I ASSUME kernel is what was initialized using he_normal
        )
    )
    model.add(
        # Originally:     Dense(256,activation='relu',init='he_normal',name="dense1")
        Dense(
            units=128, # OG 256
            activation='relu',
            kernel_initializer='he_normal' # I ASSUME kernel is what was initialized using he_normal
        )
    )

    model.add(Dropout(dr))
    model.add(
        # SM: Weird this did not come with an activation
        Dense(
            units=num_classes,
            kernel_initializer='he_normal')
    )
    model.add(Activation('softmax'))
    model.add(Reshape([num_classes]))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model


if __name__ == "__main__":
    model = build_model()