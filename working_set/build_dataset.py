#! /usr/bin/python3
import numpy as np
import tensorflow as tf
import sys

# From the homework, we used the following
# Input: 4+D tensor with shape: batch_shape + (channels, rows, cols)
# (1,2,N) Where N is the number of complex samples


# Now let's read this fuckin dataset
def build_dataset(filenames):
    if len(filenames) == 0:
        raise Exception("Hey, I need some filenames!")
    print("Parsing from: ", filenames)

    """
    This dataset contains essentially a series of dictionaries, of the following structure:
    feature = {
        'samples': _bytes_feature(tf.io.serialize_tensor(tf.constant(feature0))),
        'device_id': _int64_feature(feature1),
        'transmission_id': _int64_feature(feature2),
        'slice_index': _int64_feature(feature3),
    }

    See write_dataset.py
    """
    
    # Note how weird this is, we are generating tensors that consist only of the filenames. We then
    # interleave on that!
    filenames_ds = tf.data.Dataset.from_tensor_slices(filenames)

    # I'm trying to get some interleaving in the files we read
    # This transformation will apply map_func to cycle_length input elements (map_func returns a dataset)
    # Open iterators on the returned Dataset objects
    # Cycle through them producing block_length consecutive elements from each iterator
    # Consume the next input element each time it reaches the end of an iterator.

    # Steve Mackey:
    # By my explorations: 
    # map_func returns a dataset, which is built from the input arg. The input arg being a single element from the source dataset
    # We build <cycle_length> iterators, with each iterator being built from its own input element (it's not like they get grouped up). 
    #     In other words, its a 1:1 from input item and output dataset
    # Let iters=those iterators
    # for i in iters: return i.take(block_length)
    # Now, how does it work when an iterator gets exhausted? I have no idea, and don't care at this point.

    dataset = filenames_ds.interleave(
        lambda x: tf.data.TFRecordDataset(x),
        cycle_length=len(filenames),
        block_length=1)

    
    """
    We need to deserialize these samples.
    Deserialize into the original tensor, and device id (we drop the rest for now)
    """
    def deserialize_example(raw_record):
      example = tf.train.Example()
      example.ParseFromString(raw_record.numpy())

      tens = tf.io.parse_tensor(example.features.feature["samples"].bytes_list.value[0], np.float32)

      return [tens, example.features.feature["device_id"].int64_list.value[0]]

    """
    Wrapper to use our deserialize_example in TF
    """
    def tf_deserialize_example(raw_record):
      return tf.py_function(
        deserialize_example,
        (raw_record,),  # pass these args to the above function.
        [tf.float32, tf.int64])      # [samples], device_id

    dataset = dataset.map(tf_deserialize_example)

    """
    So now we need to get a little tricky.
    Since I'm not good with tensorflow, I am using a generator in order to take in a single sample, and then spit out a bunch
    of windowed data.
    This is likely a tensorflow sin, and I imagine there is a builtin function for this, but I can't find it
    """
    def sliding_window_generator(parsed_ds, window_size):
        for flat_time_domain_iq,device_id in parsed_ds:
            # Yeah shit breaks if they're equal, so we complain about that as well
            if window_size >= flat_time_domain_iq.shape[0]/2:
                print("Window size is >= number of IQ pairs")
                print("Window: ", window_size, "num pairs:", flat_time_domain_iq.shape[0]/2)
                raise Exception("Dun goofed on the window size")
            window_start=0
            window_end=window_size*2

            while window_end < len(flat_time_domain_iq):
                X_copy = flat_time_domain_iq.numpy().tobytes()
                sliced_X = X_copy[window_start*4:window_end*4]

                window = np.ndarray([2,int(window_size)], dtype=np.float32, buffer=sliced_X, strides=(4,8))

                yield window, device_id 

                window_start += 1
                window_end   += 1

        raise StopIteration

    final_dataset=dataset

    """
    final_dataset = tf.data.Dataset.from_generator(
        lambda: sliding_window_generator(dataset, _window_size),
        (tf.float32, tf.int64),
        (tf.TensorShape([2,_window_size]), tf.TensorShape([]))
    )
    
    final_dataset = final_dataset.shuffle(10000)
    final_dataset = final_dataset.prefetch(100)
    """

    return final_dataset

def get_num_elements_in_dataset(ds):
    count = 0
    for _ in ds:
        count += 1
    return count


if __name__ == "__main__":
    # AGH FUCK WHAT
    """
    filenames = [10,11,12,13]
    filename_ds = tf.data.Dataset.from_tensor_slices(filenames)
    
    print("OG Dataset cardinality: ", filename_ds.cardinality())

    def parse_fn(filename):
        print(filename) 
        return tf.data.Dataset.range(10)


    dataset = filename_ds.interleave(
        lambda x: parse_fn(x),
        cycle_length=4, 
        block_length=1
    )

    for x in dataset:
        print(x)
    
    print("Quit early")
    sys.exit(0) 
    """
    filenames = sys.argv[1:]

    ds = build_dataset(sys.argv[1:])

    for x,y in ds:
        print(y)
    print(get_num_elements_in_dataset(ds))
