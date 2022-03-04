import tensorflow as tf
import os 

# print the first example of the tfrecord
def print_example(tfrecord_path):
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    for example in dataset.take(1):
        print(example)
        break

x = os.path.abspath('./00-148.tfrec')
print_example(x)
