'''
    VGG16 for cifar10
    using tensorflow 2.0 alpha
'''
import tensorflow as tf
import argparse
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Conv2D, Dropout, MaxPool2D, Flatten, Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
import os

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--check', type=str, default='model/vgg1')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--drop_prob', type=float, default=0.5)
parser.add_argument('--per_save', type=int, default=10)
parser.add_argument('--mode', type=str, default='train')

params = parser.parse_args()

IMAGE_SIZE = 32
IMAGE_CHANNEL = 3
CLASSIFY = 10


def single_example_parser(serialized_example):
    features_parsed = tf.io.parse_single_example(
        serialized=serialized_example,
        features={
            'label': tf.io.FixedLenFeature([], tf.int64),
            'image': tf.io.FixedLenFeature([], tf.string)
        }
    )

    label = features_parsed['label']
    image = features_parsed['image']

    label = tf.cast(label, tf.int32)
    image = tf.io.decode_raw(image, tf.float64)
    image = tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL])
    image = (tf.cast(image, tf.float32) - 127.5) / 127.5

    return image, label


def batched_data(tfrecord_filename, single_example_parser, batch_size, buffer_size=1000, shuffle=True):
    dataset = tf.data.TFRecordDataset(tfrecord_filename)
    if shuffle:
        dataset = dataset.shuffle(buffer_size)
    dataset = dataset.map(single_example_parser) \
        .batch(batch_size) \
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


class VGG16(Model):
    def __init__(self):
        super(VGG16, self).__init__()

        self.vgg16 = Sequential([
            Conv2D(16, 3, padding='same', activation='relu'),
            Conv2D(16, 3, padding='same', activation='relu'),
            Conv2D(32, 3, padding='same', activation='relu'),
            Dropout(params.drop_prob),
            Conv2D(32, 3, padding='same', activation='relu'),
            Dropout(params.drop_prob),
            MaxPool2D(2, 2, 'same'),
            Conv2D(64, 3, padding='same', activation='relu'),
            Dropout(params.drop_prob),
            Conv2D(32, 3, padding='same', activation='relu'),
            Dropout(params.drop_prob),
            Conv2D(128, 3, padding='same', activation='relu'),
            Dropout(params.drop_prob),
            Conv2D(128, 3, padding='same', activation='relu'),
            Dropout(params.drop_prob),
            Conv2D(128, 3, padding='same', activation='relu'),
            Dropout(params.drop_prob),
            MaxPool2D(2, 2, 'same'),
            Flatten(),
            Dense(1024, activation='relu'),
            Dropout(params.drop_prob),
            Dense(1024, activation='relu'),
            Dropout(params.drop_prob),
            Dense(CLASSIFY)
        ])

    def __call__(self, image, training):
        return self.vgg16(image, training)


@tf.function
def train_step(image, label, vgg, optim, lossobj):
    with tf.GradientTape() as tape:
        logits = vgg(image, True)
        loss = lossobj(label, logits)
        acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, axis=-1, output_type=tf.int32), label), tf.float32))
    trainable_variables = vgg.trainable_variables
    gradients = tape.gradient(loss, trainable_variables)
    optim.apply_gradients(zip(gradients, trainable_variables))

    return loss, acc


def predict_step(image, label, vgg):
    logits = vgg(image, False)
    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, axis=-1), label), tf.float32))
    return acc


class USR():
    @staticmethod
    def train():
        train_file = ['data/train_cifar10.tfrecord']
        val_file = ['data/val_cifar10.tfrecord']

        valid_batch = batched_data(val_file, single_example_parser, 100, shuffle=False)
        image_val, label_val = next(iter(valid_batch))

        loss_object = SparseCategoricalCrossentropy(from_logits=True)
        optimizer = Adam(learning_rate=params.lr)

        vgg16 = VGG16()

        if not os.path.exists(params.check):
            os.makedirs(params.check)

        checkpoint_dir = params.check
        checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                         model=vgg16)
        manager = tf.train.CheckpointManager(
            checkpoint, directory=checkpoint_dir, max_to_keep=1)

        if params.mode == 'train1':
            checkpoint.restore(manager.latest_checkpoint)

        loss = []
        acc = []
        for epoch in range(1, params.epochs + 1):
            loss_epoch = []
            acc_epoch = []
            batch_data = batched_data(train_file, single_example_parser, params.batch_size,
                                      buffer_size=10 * params.batch_size)

            for batch, (image, label) in enumerate(batch_data):
                loss_batch, acc_batch = train_step(image, label, vgg16, optimizer, loss_object)
                print('>> %d  %d | loss: %f  acc: %.2f%%' % (epoch, batch + 1, loss_batch, 100.0 * acc_batch))
                loss_epoch.append(loss_batch)
                acc_epoch.append(acc_batch)

            loss.append(tf.reduce_mean(loss_epoch))
            acc.append(tf.reduce_mean(acc_epoch))
            acc_val = predict_step(image_val, label_val, vgg16)
            print('>> %d | acc_avg: %.1f%%' % (epoch, 100.0 * acc_val))

            if epoch % params.per_save == 0:
                manager.save()

    @staticmethod
    def predict():
        pass


def main():
    usr = USR()
    if params.mode.startswith('train'):
        usr.train()
    elif params.mode == 'predict':
        usr.predict()


if __name__ == '__main__':
    main()
