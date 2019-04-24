'''
    mnist data to tfrecord file
'''
import tensorflow as tf
from tensorflow.keras import datasets
import numpy as np


def write2tfrecord():
    (image, label), (_, _) = datasets.mnist.load_data()
    image = image.astype(np.float32) / 255.0
    label = label
    tfrecord_name = 'data/' + 'mnist.tfrecord'
    writer = tf.io.TFRecordWriter(tfrecord_name)
    for i in range(len(image)):
        arr_raw = image[i].tobytes()
        label_raw = label[i]
        example = tf.train.Example(
            features=tf.train.Features(feature={
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label_raw])),
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[arr_raw]))
            }))
        writer.write(example.SerializeToString())  # 序列化为字符串

    writer.close()


if __name__ == '__main__':
    write2tfrecord()
