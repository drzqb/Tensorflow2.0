'''
    DCGAN for beauty
'''
import tensorflow as tf
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tensorflow.data.experimental import AUTOTUNE
import os, sys

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--check', type=str, default='model/gan2/', help='The path where model shall be saved')
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

IMAGE_SIZE = 96
IMAGE_CHANNEL = 3


def single_example_parser(serialized_example):
    features_parsed = tf.io.parse_single_example(
        serialized=serialized_example,
        features={
            'image': tf.io.FixedLenFeature([], tf.string)
        }
    )

    image = features_parsed['image']

    image = tf.io.decode_raw(image, tf.uint8)
    image = tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL])
    image = tf.cast(image, tf.float32) / 127.5 - 1

    return image


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
        fig = plt.figure(figsize=(5, 5))
        gs = gridspec.GridSpec(5, 5)
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample)

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

        self.generator = tf.keras.Sequential([
            tf.keras.layers.Dense(8 * IMAGE_SIZE * IMAGE_SIZE),
            tf.keras.layers.Reshape([IMAGE_SIZE // 2, IMAGE_SIZE // 2, 32]),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2DTranspose(IMAGE_CHANNEL, 5, 2, 'same', activation='tanh')
        ])

    def __call__(self, noise, training=True):
        return self.generator(noise, training=training)


class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.discriminator = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 4, 2, 'same'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(100),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(1)
        ])

    def __call__(self, x, training=True):
        return tf.squeeze(self.discriminator(x, training=training), axis=-1)


@tf.function
def train_step(noise, image, generator, discriminator, loss_obj, optim_g, optim_d):
    # train discriminator once and generator twice
    with tf.GradientTape() as tape:
        fake = discriminator(generator(noise))
        real = discriminator(image)
        loss_d = loss_obj(tf.zeros_like(fake), fake) + loss_obj(tf.ones_like(real), real)

    trainable_variables_d = discriminator.trainable_variables
    gradient_d = tape.gradient(loss_d, trainable_variables_d)
    optim_d.apply_gradients(zip(gradient_d, trainable_variables_d))

    for _ in range(2):
        with tf.GradientTape() as tape:
            fake = discriminator(generator(noise))
            loss_g = loss_obj(tf.ones_like(fake), fake)

        trainable_variables_g = generator.trainable_variables
        gradient_g = tape.gradient(loss_g, trainable_variables_g)
        optim_g.apply_gradients(zip(gradient_g, trainable_variables_g))

    return loss_g, loss_d


def predict_step(noise, generator):
    return generator(noise, False)


class USR():
    @staticmethod
    def train():
        train_file = ['data/beauty.tfrecord']

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
            for batch, image in enumerate(
                    batched_data(train_file, single_example_parser, params.batch_size, 10 * params.batch_size)):
                noise = Data.sample_Z(params.batch_size, params.noise_dim)

                loss_g, loss_d = train_step(noise, image, generator, discriminator, loss_obj, optimizer_g, optimizer_d)

                sys.stdout.write('\r>> Epoch %d  Batch:%d | loss_g: %.9f  loss_d: %.9f' % (
                    epoch, batch + 1, loss_g, loss_d))
                sys.stdout.flush()

            sys.stdout.write('\n')
            sys.stdout.flush()

            samples = predict_step(Data.sample_Z(25, params.noise_dim), generator)
            Plot.plot_image(tf.cast(tf.clip_by_value((samples + 1) * 127.5, 0, 255), tf.uint8), str(epoch))

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
