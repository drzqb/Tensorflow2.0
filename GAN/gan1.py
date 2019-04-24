'''
    CGAN for mnist
'''
import tensorflow as tf
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tensorflow.data.experimental import AUTOTUNE
import os, sys

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--check', type=str, default='model/gan1/', help='The path where model shall be saved')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size during training')
parser.add_argument('--noise_dim', type=int, default=100)
parser.add_argument('--epochs', type=int, default=1000, help='Epochs during training')
parser.add_argument('--lr_d', type=float, default=0.0002)
parser.add_argument('--lr_g', type=float, default=0.0002)
parser.add_argument('--beta_d', type=float, default=0.5)
parser.add_argument('--beta_g', type=float, default=0.5)
parser.add_argument('--mode', type=str, default='train0',
                    help='The mode of train or predict as follows: '
                         'train0: begin to train or retrain'
                         'tran1:continue to train'
                         'predict: predict')
parser.add_argument('--per_save', type=int, default=10, help='save model for every per_save')

params = parser.parse_args()

IMAGE_SIZE = 28
IMAGE_SIZE2 = IMAGE_SIZE ** 2
N_CLASS = 10


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
    image = tf.reshape(image, [IMAGE_SIZE2])
    return image, label


def batched_data(tfrecord_filename, single_example_parser, batch_size, buffer_size=1000, shuffle=True):
    dataset = tf.data.TFRecordDataset(tfrecord_filename)
    if shuffle:
        dataset = dataset.shuffle(buffer_size)
    dataset = dataset.map(single_example_parser) \
        .batch(batch_size, drop_remainder=True) \
        .prefetch(buffer_size=AUTOTUNE)

    return dataset


class Data():
    @staticmethod
    def sample_Z(m, n):
        return np.random.uniform(-1., 1., size=[m, n]).astype(np.float32)


class Plot():
    @staticmethod
    def plot_image(samples, suffix=None):
        fig = plt.figure(figsize=(10, 6))
        gs = gridspec.GridSpec(5, 10)
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_aspect('equal')
            plt.imshow(tf.reshape(sample,[IMAGE_SIZE, IMAGE_SIZE]))

        if suffix is not None:
            plt.savefig(params.check + suffix + '.png', bbox_inches='tight')
        else:
            plt.savefig(params.check + 'predict.png', bbox_inches='tight')

        plt.close(fig)

    @staticmethod
    def plot_loss(g_loss, d_loss):
        plt.plot(g_loss, label='G_Loss')
        plt.plot(d_loss, label='D_Loss')
        plt.legend(loc='upper right')
        plt.savefig(params.check + 'Loss.png', bbox_inches='tight')
        plt.close()


class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()

        self.dense_noise = tf.keras.layers.Dense(128, activation='relu')
        self.dense_c = tf.keras.layers.Dense(128, activation='relu')
        self.dense_output = tf.keras.layers.Dense(IMAGE_SIZE2, activation='sigmoid')

    def __call__(self, noise, c):
        dense_noise = self.dense_noise(noise)
        dense_c = self.dense_c(c)
        return self.dense_output(tf.concat([dense_noise, dense_c], axis=-1))


class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.dense = tf.keras.layers.Dense(128, activation='relu')
        self.dense_output = tf.keras.layers.Dense(1, activation='sigmoid')

    def __call__(self, x, c):
        xc = self.dense(tf.concat([x, c], axis=-1))
        return tf.squeeze(self.dense_output(xc), axis=-1)


@tf.function
def train_step(noise, image, label, generator, discriminator, loss_obj, optim_g, optim_d):
    # train discriminator once and generator twice
    with tf.GradientTape() as tape:
        fake = discriminator(generator(noise, label), label)
        real = discriminator(image, label)
        loss_d = loss_obj(tf.zeros_like(fake), fake) + loss_obj(tf.ones_like(real), real)

    trainable_variables_d = discriminator.trainable_variables
    gradient_d = tape.gradient(loss_d, trainable_variables_d)
    optim_d.apply_gradients(zip(gradient_d, trainable_variables_d))

    for _ in range(2):
        with tf.GradientTape() as tape:
            fake = discriminator(generator(noise, label), label)
            loss_g = loss_obj(tf.ones_like(fake), fake)

        trainable_variables_g = generator.trainable_variables
        gradient_g = tape.gradient(loss_g, trainable_variables_g)
        optim_g.apply_gradients(zip(gradient_g, trainable_variables_g))

    return loss_g, loss_d


def predict_step(noise, label, generator):
    return generator(noise, label)


class USR():
    @staticmethod
    def train():
        train_file = ['data/mnist.tfrecord']

        generator = Generator()
        discriminator = Discriminator()
        optimizer_g = tf.keras.optimizers.Adam(params.lr_g, beta_1=params.beta_g)
        optimizer_d = tf.keras.optimizers.Adam(params.lr_d, beta_1=params.beta_d)
        loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        if not os.path.exists(params.check):
            os.makedirs(params.check)

        checkpoint_dir = params.check
        checkpoint = tf.train.Checkpoint(optimizer_g=optimizer_g,
                                         optimizer_d=optimizer_d,
                                         generator=generator,
                                         discriminator=discriminator
                                         )
        manager = tf.train.CheckpointManager(
            checkpoint, directory=checkpoint_dir, max_to_keep=1)

        if params.mode == 'train1':
            checkpoint.restore(manager.latest_checkpoint)

        for epoch in range(1, params.epochs + 1):
            for batch, (image, label) in enumerate(
                    batched_data(train_file, single_example_parser, params.batch_size, 10 * params.batch_size)):
                noise = Data.sample_Z(params.batch_size, params.noise_dim)

                loss_g, loss_d = train_step(noise, image, tf.one_hot(label, N_CLASS), generator, discriminator,
                                            loss_obj, optimizer_g, optimizer_d)

                sys.stdout.write('\r>> Epoch %d  Batch:%d | loss_g: %.9f  loss_d: %.9f' % (
                    epoch, batch + 1, loss_g, loss_d))
                sys.stdout.flush()

            sys.stdout.write('\n')
            sys.stdout.flush()

            samples = predict_step(Data.sample_Z(50, params.noise_dim),
                                   tf.one_hot(np.tile([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [5]), N_CLASS), generator)
            Plot.plot_image(samples, str(epoch))

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
