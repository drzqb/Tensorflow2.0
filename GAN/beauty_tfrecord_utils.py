'''
    beauty image data to tfrecord file
'''
import tensorflow as tf
import os
from PIL import Image
import numpy as np


def write2tfrecord(data_path, suffix):
    tfrecord_name = 'data/' + suffix + '.tfrecord'
    writer = tf.io.TFRecordWriter(tfrecord_name)
    files = os.listdir(data_path)
    r = np.random.permutation(len(files))
    files = [files[i] for i in r]
    for image_filename in files:
        if image_filename.endswith('.jpg'):
            arr = Image.open(os.path.join(data_path, image_filename))
            arr_raw = arr.tobytes()
            example = tf.train.Example(
                features=tf.train.Features(feature={
                    'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[arr_raw]))
                }))
            writer.write(example.SerializeToString())  # 序列化为字符串

    writer.close()


if __name__ == '__main__':
    write2tfrecord('data/faces/', 'beauty')
