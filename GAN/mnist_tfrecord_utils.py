'''
    mnist data to tfrecord file
'''
import tensorflow as tf
from tensorflow.data.experimental import AUTOTUNE
from tensorflow.keras import datasets
import matplotlib.pyplot as plt
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


def single_example_parser(serialized_example):
    features_parsed = tf.io.parse_single_example(
        serialized=serialized_example,
        features={
            'label': tf.io.FixedLenFeature([], tf.int64),
            'image': tf.io.FixedLenFeature([], tf.string)
        }
    )

    image = features_parsed['image']
    label = features_parsed['label']

    image = tf.io.decode_raw(image, tf.float32)
    image = tf.reshape(image, [28, 28, 1])
    return image, label


def batched_data(tfrecord_filename, single_example_parser, batch_size, buffer_size=1000, shuffle=True):
    dataset = tf.data.TFRecordDataset(tfrecord_filename)
    if shuffle:
        dataset = dataset.shuffle(buffer_size)
    dataset = dataset.map(single_example_parser) \
        .batch(batch_size, drop_remainder=True) \
        .prefetch(buffer_size=AUTOTUNE)

    return dataset


if __name__ == '__main__':
    write2tfrecord()

    # dataset = batched_data(['data/mnist.tfrecord'], single_example_parser, 7)
    # print(dataset)
    # dataset=iter(dataset)
    # print(dataset)
    # print(next(dataset)[1])
    # print(next(dataset)[1])
    # print(next(dataset)[1])
    # print(next(dataset)[1])
    # for i,(image,label) in enumerate(dataset.take(10)):
    #     print(label)
    #
    # for i,(image,label) in enumerate(dataset.take(10)):
    #     print(label)
