'''
  隐语义模型 LFM (Latent Factor Model)
  R=PQ^T        P: 用户因子矩阵   Q: 物品因子矩阵
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
parser.add_argument('--embedding_size', type=int, default=16)
parser.add_argument('--check', type=str, default='model/lfm2')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--drop_prob', type=float, default=0.2)
parser.add_argument('--per_save', type=int, default=10)
parser.add_argument('--mode', type=str, default='train0')

params = parser.parse_args()


class LFM(tf.keras.Model):
    def __init__(self, n_user, n_item):
        super(LFM, self).__init__()

        self.user_embed = tf.keras.layers.Embedding(n_user, params.embedding_size)
        self.user_dropout = tf.keras.layers.Dropout(params.drop_prob)
        self.item_embed = tf.keras.layers.Embedding(n_item, params.embedding_size)
        self.item_dropout = tf.keras.layers.Dropout(params.drop_prob)

    def __call__(self, user, item, training):
        user_embed = self.user_embed(user)
        user_dropout = self.user_dropout(user_embed, training=training)
        item_embed = self.item_embed(item)
        item_dropout = self.item_dropout(item_embed, training=training)
        return tf.reduce_sum(tf.multiply(user_dropout, item_dropout), axis=-1)


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
        lfm = LFM(n_user + 1, n_item + 1)

        if not os.path.exists(params.check):
            os.makedirs(params.check)

        checkpoint_dir = params.check
        checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                         model=lfm)
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
                loss_batch = train_step(user, item, rating, lfm, loss_object, optimizer)
                sys.stdout.write('\r>> %d  %d | loss: %f' % (epoch, batch + 1, loss_batch))
                sys.stdout.flush()
                loss_epoch.append(loss_batch)

            loss.append(tf.reduce_mean(loss_epoch))

            sys.stdout.write(' | loss_avg: %f' % (loss[-1]))
            sys.stdout.flush()

            _, loss_val = predict_step(user_val, item_val, rating_val, lfm, loss_object)
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
