'''
    IPNN (Inner Product-based Neural Network) for CTR
    using Tensorflow 2.0 alpha
'''
import tensorflow as tf
import numpy as np
import sys, os
import matplotlib.pyplot as plt
import argparse
from build_data import load_data

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--embedding_size', type=int, default=256)
parser.add_argument('--product_size', type=int, default=128)
parser.add_argument('--hidden_size', type=int, default=[64, 32, 16])
parser.add_argument('--check', type=str, default='model/ipnn1')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--drop_prob', type=float, default=0.5)
parser.add_argument('--per_save', type=int, default=10)
parser.add_argument('--mode', type=str, default='train0')

params = parser.parse_args()


class IPNN(tf.keras.Model):
    def __init__(self, feature_size):
        super(IPNN, self).__init__()

        self.embedding = tf.keras.layers.Embedding(feature_size, params.embedding_size)
        self.flatten = tf.keras.layers.Flatten()
        self.dense_lz = tf.keras.layers.Dense(params.product_size)
        self.dense_lp = tf.keras.layers.Dense(params.product_size, use_bias=False)
        self.dense_h = [tf.keras.layers.Dense(params.hidden_size[i], activation='relu') for i in
                        range(len(params.hidden_size))]
        self.drop_h = [tf.keras.layers.Dropout(params.drop_prob) for i in range(len(params.hidden_size))]
        self.project = tf.keras.layers.Dense(1)

    def __call__(self, index, value, training):
        embedding = self.embedding(index)  # Batch*Feild*Embedding
        feature_mul_value = tf.multiply(embedding, tf.expand_dims(value, axis=2))   # Batch*Feild*Embedding
        lz = self.dense_lz(self.flatten(feature_mul_value))  # Batch*D1
        lp = self.dense_lp(tf.transpose(feature_mul_value, [0, 2, 1]))  # Batch*Embedding*D1
        lp = tf.reduce_sum(tf.square(lp), axis=1)  # Batch*D1
        now = tf.nn.relu(lp + lz)

        for i in range(len(params.hidden_size)):
            now = self.drop_h[i](self.dense_h[i](now), training=training)
        logits = self.project(now)
        return tf.squeeze(logits, axis=-1)


@tf.function
def train_step(index, value, label, model, lossobj, optim, auc):
    with tf.GradientTape() as tape:
        logits = model(index, value, True)
        loss = lossobj(label, logits)
        auc.update_state(label, tf.sigmoid(logits))
    trainable_variables = model.trainable_variables
    gradients = tape.gradient(loss, trainable_variables)
    optim.apply_gradients(zip(gradients, trainable_variables))
    return loss, auc.result()


def predict_step(index, value, label, model, lossobj, auc):
    logits = model(index, value, False)
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
        model = IPNN(feature_size)

        if not os.path.exists(params.check):
            os.makedirs(params.check)

        checkpoint_dir = params.check
        checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                         model=model)
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
                loss_batch, auc_batch = train_step(index_b, value_b, label_b, model, loss_object, optimizer, Auc)
                sys.stdout.write('\r>> %d  %d | loss: %f  auc: %.2f' % (epoch, batch + 1, loss_batch, auc_batch))
                sys.stdout.flush()
                loss_epoch.append(loss_batch)
                auc_epoch.append(auc_batch)

            loss.append(tf.reduce_mean(loss_epoch))
            auc.append(tf.reduce_mean(auc_epoch))

            sys.stdout.write(' | loss_avg: %f  auc_avg: %.2f' % (loss[-1], auc[-1]))
            sys.stdout.flush()

            loss_val, auc_val = predict_step(index_val, value_val, label_val, model, loss_object, Auc)
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
