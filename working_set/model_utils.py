#! /usr/bin/python3
import sys, os

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Reshape,Dense,Dropout,Activation,Flatten, Convolution1D, MaxPooling2D, ZeroPadding2D, Permute
import tensorflow.keras.models as models
import tensorflow.keras as keras

import matplotlib.pyplot as plt

from build_dataset import build_dataset,get_num_elements_in_dataset, build_debug_set

NUM_EPOCH=100

if NUM_EPOCH < 100:
    print("AGGGGGGGGGGGGGGGGGGGGHHHHHHHHHHHHHHHHHHHHHHHHHHHH")
    print("RUNNING WITH SUPER LOW EPOCHS!")
    print("AGGGGGGGGGGGGGGGGGGGGHHHHHHHHHHHHHHHHHHHHHHHHHHHH")

def configure_and_split_dataset(dataset, ds_size, ds_cache_path):
    TRAIN_RATIO=0.7
    VALIDATION_RATIO=0.2
    TEST_RATIO=0.1
    PREFETCH_SIZE=1000
    BATCH_SIZE=250

    # Caching needs to be in this order for shuffling

    dataset = dataset.cache(filename=ds_cache_path)
    dataset = dataset.shuffle(ds_size, seed=1337, reshuffle_each_iteration=False)
    
    train_dataset_size = int(ds_size * TRAIN_RATIO)
    val_dataset_size   = int(ds_size * VALIDATION_RATIO)
    test_dataset_size  = int(ds_size * TEST_RATIO)

    train_dataset = dataset.take(train_dataset_size)
    # train_dataset = train_dataset.cache(filename=workspace_path+"/"+TRAIN_CACHE_NAME)
    train_dataset = train_dataset.prefetch(PREFETCH_SIZE)
    train_dataset = train_dataset.batch(BATCH_SIZE)

    val_dataset = dataset.skip(train_dataset_size).take(val_dataset_size)
    # val_dataset = val_dataset.cache(filename=workspace_path+"/"+VAL_CACHE_NAME)
    val_dataset = val_dataset.prefetch(PREFETCH_SIZE)
    val_dataset = val_dataset.batch(BATCH_SIZE)

    test_dataset = dataset.skip(train_dataset_size+val_dataset_size).take(test_dataset_size)
    # test_dataset = test_dataset.cache(filename=workspace_path+"/"+TEST_CACHE_NAME)
    test_dataset = test_dataset.prefetch(PREFETCH_SIZE)
    test_dataset = test_dataset.batch(BATCH_SIZE)

    return {"train": train_dataset, "val": val_dataset, "test":test_dataset}

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


# build_dataset_func takes a list of paths or tuples
# This is clearly a shit abstraction lmao
def ds_config_to_datasets(ds_config, build_dataset_func):
    CACHE_PATH = "_cache"

    dataset = build_dataset_func(ds_config["train_paths"] + ds_config["test_paths"])

    if "size" not in ds_config.keys()  or ds_config["size"] == None:
        print("Num elements in dataset: ", get_num_elements_in_dataset(dataset))
        print("Add this to the ds_config, then rerun")
        sys.exit(1)

    return configure_and_split_dataset(dataset, ds_config["size"], os.path.join(CACHE_PATH, ds_config["cache_name"]))
    

    

def train_model(model, train_ds, val_ds, weights_path, num_epochs=None):
    num_epochs = (num_epochs if num_epochs != None else NUM_EPOCH)

    history = model.fit(
        train_ds,
        #batch_size=batch_size,
        epochs=num_epochs,
        verbose=2,
        validation_data=val_ds,
        callbacks = [
            keras.callbacks.ModelCheckpoint(weights_path, monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
        ]
    )


def test_model(model, test_ds, min_class_label, max_class_label):
    num_classes = max_class_label - min_class_label + 1
    test_Y_hat = model.predict(test_ds)
    conf = np.zeros([num_classes,num_classes])
    confnorm = np.zeros([num_classes,num_classes])

    counter = 0
    correct = 0
    incorrect = 0
    for x,y in test_ds.unbatch(): # Batching was making this weird
        
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

    plot_confusion_matrix(confnorm, labels=list(range(min_class_label, max_class_label+1)))
    # plt.savefig("confusion.png")
    #plt.figure()