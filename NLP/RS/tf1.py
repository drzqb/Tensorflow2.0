'''
    Transformer for CTR
    using tf 2.0 alpha
'''
import numpy as np
import tensorflow as tf
import sys, os
from build_data import load_data
import argparse
import matplotlib.pylab as plt

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--embedding_size', type=int, default=32)
parser.add_argument('--block', type=int, default=2)
parser.add_argument('--head', type=int, default=4)
parser.add_argument('--check', type=str, default='model/tf1')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--per_save', type=int, default=10)
parser.add_argument('--mode', type=str, default='train')

params = parser.parse_args()


class Attention(tf.keras.layers.Layer):
    def __init__(self):
        super(Attention, self).__init__()

        self.dense_q = tf.keras.layers.Dense(params.embedding_size, use_bias=False)
        self.dense_k = tf.keras.layers.Dense(params.embedding_size, use_bias=False)
        self.dense_v = tf.keras.layers.Dense(params.embedding_size, use_bias=False)
        self.dense_o = tf.keras.layers.Dense(params.embedding_size, use_bias=False)

    def __call__(self, x):
        Q = tf.concat(tf.split(self.dense_q(x), params.head, axis=-1), axis=0)
        K = tf.concat(tf.split(self.dense_k(x), params.head, axis=-1), axis=0)
        V = tf.concat(tf.split(self.dense_v(x), params.head, axis=-1), axis=0)
        QK = tf.matmul(Q, tf.transpose(K, [0, 2, 1])) / tf.sqrt(params.embedding_size / params.head)
        return self.dense_o(tf.concat(tf.split(tf.matmul(tf.nn.softmax(QK, axis=-1), V), params.head, axis=0), axis=-1))


class FeedForward(tf.keras.layers.Layer):
    def __init__(self):
        super(FeedForward, self).__init__()

        self.dense_ffrelu = tf.keras.layers.Dense(2 * params.embedding_size, activation='relu')
        self.dense_ff = tf.keras.layers.Dense(params.embedding_size)

    def __call__(self, x):
        return self.dense_ff(self.dense_ffrelu(x))


class AddAndNormalize(tf.keras.layers.Layer):
    def __init__(self):
        super(AddAndNormalize, self).__init__()

        self.layernorm = tf.keras.layers.experimental.LayerNormalization(epsilon=1e-8)

    def __call__(self, x, y):
        return self.layernorm(x + y)


class Transformer(tf.keras.Model):
    def __init__(self, feature_size):
        super(Transformer, self).__init__()

        self.embedding1 = tf.keras.layers.Embedding(feature_size, 1)
        self.embedding2 = tf.keras.layers.Embedding(feature_size, params.embedding_size)
        self.attention = [Attention() for _ in range(params.block)]
        self.addlayernorm = [AddAndNormalize() for _ in range(2 * params.block)]
        self.feedforward = [FeedForward() for _ in range(params.block)]
        self.flatten = tf.keras.layers.Flatten()
        self.out_dense = tf.keras.layers.Dense(1)

    def __call__(self, feature_index, feature_value, label):
        embedding1 = self.embedding1(feature_index)
        embedding2 = self.embedding2(feature_index)

        output1 = tf.reduce_sum(tf.squeeze(embedding1, axis=-1), axis=-1, keepdims=True)
        now = tf.multiply(embedding2, tf.expand_dims(feature_value, axis=2))

        for i in range(params.block):
            now = self.addlayernorm[2 * i](now, self.attention[i](now))
            now = self.addlayernorm[2 * i + 1](now, self.feedforward[i](now))

        return tf.squeeze(output1 + self.out_dense(self.flatten(now)), axis=-1)


@tf.function
def train_step(index, value, label, transformer, lossobj, optim, auc):
    with tf.GradientTape() as tape:
        logits = transformer(index, value, label)
        loss = lossobj(label, logits)
        auc.update_state(label, tf.sigmoid(logits))
    trainable_variables = transformer.trainable_variables
    gradients = tape.gradient(loss, trainable_variables)
    optim.apply_gradients(zip(gradients, trainable_variables))
    return loss, auc.result()


def predict_step(index, value, label, transformer: Transformer, lossobj, auc):
    logits = transformer(index, value, label)
    loss = lossobj(label, logits)
    auc.update_state(label, tf.sigmoid(logits))

    return loss, auc.result()


class USR():
    @staticmethod
    def train():
        data = load_data()

        feature_size = data['feat_dim']

        index = np.array(data['xi'], dtype=np.int32)
        value = np.array(data['xv'], dtype=np.float32)
        label = np.squeeze(np.array(data['y_train'], dtype=np.int32), axis=-1)

        train_dataset = tf.data.Dataset.from_tensor_slices(
            (index[params.batch_size:], value[params.batch_size:], label[params.batch_size:]))

        valid_dataset = tf.data.Dataset.from_tensor_slices(
            (index[:params.batch_size], value[:params.batch_size], label[:params.batch_size])) \
            .batch(params.batch_size)

        index_val, value_val, label_val = next(iter(valid_dataset))

        loss_object = tf.losses.BinaryCrossentropy(from_logits=True)
        optimizer = tf.optimizers.Adam(learning_rate=params.lr, clipnorm=5.0)
        Auc = tf.keras.metrics.AUC()
        transformer = Transformer(feature_size)

        if not os.path.exists(params.check):
            os.makedirs(params.check)

        checkpoint_dir = params.check
        checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                         model=transformer)
        manager = tf.train.CheckpointManager(
            checkpoint, directory=checkpoint_dir, max_to_keep=1)

        if params.mode == 'train1':
            checkpoint.restore(manager.latest_checkpoint)

        loss = []
        auc = []
        for epoch in range(1, params.epochs + 1):
            loss_epoch = []
            auc_epoch = []

            for batch, (index_b, value_b, label_b) in enumerate(
                    train_dataset.shuffle(buffer_size=10 * params.batch_size) \
                            .batch(params.batch_size, drop_remainder=True) \
                            .prefetch(tf.data.experimental.AUTOTUNE)):
                loss_batch, auc_batch = train_step(index_b, value_b, label_b, transformer, loss_object, optimizer, Auc)
                sys.stdout.write('\r>> %d  %d | loss: %f  auc: %.2f' % (epoch, batch + 1, loss_batch, auc_batch))
                sys.stdout.flush()
                loss_epoch.append(loss_batch)
                auc_epoch.append(auc_batch)

            loss.append(tf.reduce_mean(loss_epoch))
            auc.append(tf.reduce_mean(auc_epoch))

            sys.stdout.write(' | loss_avg: %f  auc_avg: %.2f' % (loss[-1], auc[-1]))
            sys.stdout.flush()

            loss_val, auc_val = predict_step(index_val, value_val, label_val, transformer, loss_object, Auc)
            sys.stdout.write('  | loss_val: %f  auc_val: %.2f\n' % (loss_val, auc_val))
            sys.stdout.flush()

            if epoch % params.per_save == 0:
                manager.save()

        fig = plt.figure()

        ax1 = fig.add_subplot(111)
        l1 = ax1.plot(loss, 'b', label='loss')
        ax1.set_ylabel('Loss')
        ax2 = ax1.twinx()
        l2 = ax2.plot(auc, 'r', label='auc')
        ax2.set_ylabel('AUC')
        ax2.set_xlabel('Epoch')
        ax2.set_title('Loss and AUC')

        ls = l1 + l2
        labels = [l.get_label() for l in ls]
        ax2.legend(ls, labels, loc='best')
        plt.show()

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
