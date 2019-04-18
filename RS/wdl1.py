'''
  Wide and Deep Learning
  基于TensorFlow 2.0 alpha
'''
import tensorflow as tf
import pandas as pd
import numpy as np
import argparse
import matplotlib.pylab as plt
import os, sys

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--embedding_size', type=int, default=128)
parser.add_argument('--check', type=str, default='model/wdl1')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--drop_prob', type=float, default=0.2)
parser.add_argument('--per_save', type=int, default=10)
parser.add_argument('--mode', type=str, default='train0')

params = parser.parse_args()


class WDL(tf.keras.Model):
    def __init__(self, n_user, n_item):
        super(WDL, self).__init__()

        self.user_embed = tf.keras.layers.Embedding(n_user, params.embedding_size)
        self.item_embed = tf.keras.layers.Embedding(n_item, params.embedding_size)
        self.dropout = [tf.keras.layers.Dropout(0.2),
                        tf.keras.layers.Dropout(0.5),
                        tf.keras.layers.Dropout(0.5),
                        tf.keras.layers.Dropout(0.5)]
        self.dense = [tf.keras.layers.Dense(params.embedding_size, activation='relu'),
                      tf.keras.layers.Dense(params.embedding_size / 4, activation='relu'),
                      tf.keras.layers.Dense(params.embedding_size / 16, activation='relu'),
                      tf.keras.layers.Dense(1)]

    def __call__(self, user, item, training):
        user_embed = self.user_embed(user)
        item_embed = self.item_embed(item)
        now = tf.concat([user_embed, item_embed], axis=-1)
        for i in range(4):
            now = self.dense[i](self.dropout[i](now, training=training))
        now = 5.0 * tf.sigmoid(now)
        return tf.squeeze(now, axis=-1)


@tf.function
def train_step(user, item, rating, lfm, lossobj, optim):
    with tf.GradientTape() as tape:
        logits = lfm(user, item, True)
        loss = lossobj(rating, logits)

    trainable_variables = lfm.trainable_variables
    gradients = tape.gradient(loss, trainable_variables)
    optim.apply_gradients(zip(gradients, trainable_variables))
    return loss


def predict_step(user, item, rating, lfm, lossobj):
    logits = lfm(user, item, False)
    loss = lossobj(rating, logits)
    return logits, loss


class USR():
    @staticmethod
    def train():
        ratings = pd.read_csv('data/ml-1m/ratings.dat', sep='::', names=['user_id', 'movie_id', 'rating', 'timestamp'],
                              header=None, engine='python')
        n_user = np.max(ratings['user_id'])
        n_item = np.max(ratings['movie_id'])

        user_train = np.array(ratings['user_id'], dtype=np.int32)
        item_train = np.array(ratings['movie_id'], dtype=np.int32)
        rating_train = np.array(ratings['rating'], dtype=np.float32)
        m_sample = user_train.shape[0]
        r = np.random.permutation(m_sample)
        user_train = user_train[r]
        item_train = item_train[r]
        rating_train = rating_train[r]

        data_train = tf.data.Dataset.from_tensor_slices(
            (user_train[params.batch_size:], item_train[params.batch_size:], rating_train[params.batch_size:]))

        data_val = tf.data.Dataset.from_tensor_slices(
            (user_train[:params.batch_size], item_train[:params.batch_size], rating_train[:params.batch_size])).batch(
            params.batch_size)

        user_val, item_val, rating_val = next(iter(data_val))

        loss_object = tf.losses.MeanSquaredError()
        optimizer = tf.optimizers.Adam(learning_rate=params.lr)
        wdl = WDL(n_user + 1, n_item + 1)

        if not os.path.exists(params.check):
            os.makedirs(params.check)

        checkpoint_dir = params.check
        checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                         model=wdl)
        manager = tf.train.CheckpointManager(
            checkpoint, directory=checkpoint_dir, max_to_keep=1)

        if params.mode == 'train1':
            checkpoint.restore(manager.latest_checkpoint)

        loss = []
        for epoch in range(1, params.epochs + 1):
            loss_epoch = []
            for batch, (user, item, rating) in enumerate(data_train.shuffle(buffer_size=10 * params.batch_size) \
                                                                 .batch(params.batch_size, drop_remainder=True) \
                                                                 .prefetch(tf.data.experimental.AUTOTUNE)):
                loss_batch = train_step(user, item, rating, wdl, loss_object, optimizer)
                sys.stdout.write('\r>> %d  %d | loss: %f' % (epoch, batch + 1, loss_batch))
                sys.stdout.flush()
                loss_epoch.append(loss_batch)

            loss.append(tf.reduce_mean(loss_epoch))

            sys.stdout.write(' | loss_avg: %f' % (loss[-1]))
            sys.stdout.flush()

            _, loss_val = predict_step(user_val, item_val, rating_val, wdl, loss_object)
            sys.stdout.write('  | loss_val: %f\n' % (loss_val))
            sys.stdout.flush()
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
