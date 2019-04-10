'''
    cifar10 image data to tfrecord file
'''
import tensorflow as tf
import os
import numpy as np
import pickle

IMAGE_SIZE = 32
IMAGE_CHANNEL = 3


def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='latin1')
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).astype("float")
        Y = np.array(Y)
        return X, Y


def load_CIFAR10(ROOT):
    """ load all of cifar """
    Xtr = np.zeros((50000, 3, 32, 32))
    Ytr = np.zeros(50000, dtype=np.int)
    for b in range(1, 6):
        Xtr[(b - 1) * 10000:b * 10000], Ytr[(b - 1) * 10000:b * 10000] = load_CIFAR_batch(
            os.path.join(ROOT, 'data_batch_%d' % (b)))
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return np.transpose(Xtr, [0, 2, 3, 1]), Ytr, np.transpose(Xte, [0, 2, 3, 1]), Yte


def write2tfrecord(image, label, data_usage, suffix):
    tfrecord_name = 'data/' + data_usage + '_' + suffix + '.tfrecord'
    writer = tf.io.TFRecordWriter(tfrecord_name)
    m_samples = np.shape(image)[0]
    for i in range(m_samples):
        arr_raw = image[i].tobytes()
        example = tf.train.Example(
            features=tf.train.Features(feature={
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label[i]])),
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[arr_raw]))
            }))
        writer.write(example.SerializeToString())  # 序列化为字符串

    writer.close()


def main():
    trX, trY, teX, teY = load_CIFAR10('data\cifar-10-batches-py')

    write2tfrecord(trX, trY, 'train', 'cifar10')
    write2tfrecord(teX, teY, 'val', 'cifar10')


if __name__ == '__main__':
    main()
