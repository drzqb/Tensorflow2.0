'''
    BiLSTM for NER with Tensorflow 2.0 alpha
'''
import tensorflow as tf
import numpy as np
import pickle
import os
import sys
import matplotlib.pylab as plt
import argparse
from tensorflow.keras.layers import Embedding, Dense, GRU, Bidirectional, Dropout, Flatten, Reshape
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--embedding_size', type=int, default=50)
parser.add_argument('--rnn_size', type=int, default=128)
parser.add_argument('--check', type=str, default='model/ner1')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--keep_prob', type=float, default=0.5)
parser.add_argument('--per_save', type=int, default=10)
parser.add_argument('--mode', type=str, default='train')

CONFIG = parser.parse_args()


def single_example_parser(serialized_example):
    context_features = {
        'length': tf.io.FixedLenFeature([], tf.int64)
    }
    sequence_features = {
        'sen': tf.io.FixedLenSequenceFeature([],
                                             tf.int64),
        'ner': tf.io.FixedLenSequenceFeature([],
                                             tf.int64)}

    context_parsed, sequence_parsed = tf.io.parse_single_sequence_example(
        serialized=serialized_example,
        context_features=context_features,
        sequence_features=sequence_features
    )

    length = context_parsed['length']

    sen = sequence_parsed['sen']
    ner = sequence_parsed['ner']
    return sen, ner, length


def batched_data(tfrecord_filename, single_example_parser, batch_size, padded_shapes, buffer_size=1000,
                 shuffle=True):
    dataset = tf.data.TFRecordDataset(tfrecord_filename)
    if shuffle:
        dataset = dataset.shuffle(buffer_size)
    dataset = dataset.map(single_example_parser) \
        .padded_batch(batch_size, padded_shapes=padded_shapes) \
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


class NER(tf.keras.Model):
    def __init__(self, config, word_dict_len, ner_dict_len):
        super(NER, self).__init__()

        self.config = config
        self.word_dict_len = word_dict_len
        self.ner_dict_len = ner_dict_len

        self.embedding = Embedding(self.word_dict_len, self.config.embedding_size)
        self.GRU = GRU(self.config.rnn_size, return_sequences=True)
        self.bidirectionalgru = Bidirectional(self.GRU)
        self.dense = Dense(self.ner_dict_len)

    def call(self, sentence):
        embedding = self.embedding(sentence)
        bidigru = self.bidirectionalgru(embedding)
        output = self.dense(bidigru)
        return output


@tf.function
def train_step(sen_batch, ner_batch, len_batch, ner, loss_object, optimizer):
    with tf.GradientTape() as tape:
        logits = ner(sen_batch)
        mask = tf.cast(tf.math.greater(sen_batch, 0), dtype=tf.float32)
        length = tf.cast(tf.reduce_sum(len_batch), tf.float32)
        loss_batch = loss_object(ner_batch, logits)
        loss_batch *= mask
        loss_batch = tf.reduce_sum(loss_batch) / length

        acc_batch = tf.cast(tf.equal(tf.argmax(logits, axis=-1), ner_batch), tf.float32)
        acc_batch *= mask
        acc_batch = tf.reduce_sum(acc_batch) / length
    gradients = tape.gradient(loss_batch, ner.trainable_variables)
    optimizer.apply_gradients(zip(gradients, ner.trainable_variables))

    return loss_batch, acc_batch


def val_step(sen_val, ner_val, len_val, ner, loss_object):
    logits = ner(sen_val)
    mask = tf.cast(tf.math.greater(sen_val, 0), dtype=tf.float32)
    length = tf.cast(tf.reduce_sum(len_val), tf.float32)
    loss_val = loss_object(ner_val, logits)
    loss_val *= mask
    loss_val = tf.reduce_sum(loss_val) / length

    acc_val = tf.cast(tf.equal(tf.argmax(logits, axis=-1), ner_val), tf.float32)
    acc_val *= mask
    acc_val = tf.reduce_sum(acc_val) / length

    return loss_val, acc_val


def predict_step(sen_test, ner):
    logits = ner(sen_test)
    mask = tf.cast(tf.math.greater(sen_test, 0), dtype=tf.int32)

    result = tf.argmax(logits, axis=-1, output_type=tf.int32)
    result *= mask

    return result


class USR():
    def __init__(self, config):
        self.config = config

        with open('data/word_dict.txt', 'rb') as f:
            self.word_dict = pickle.load(f)
        with open('data/ner_dict.txt', 'rb') as f:
            self.ner_dict = pickle.load(f)

    def train(self):
        train_file = ['data/train.tfrecord']
        valid_file = ['data/valid.tfrecord']

        valid_batch = batched_data(valid_file, single_example_parser, 110, padded_shapes=([-1], [-1], []),
                                   shuffle=False)
        sen_val, ner_val, len_val = next(iter(valid_batch))

        ner = NER(self.config, len(self.word_dict), len(self.ner_dict))
        optimizer = Adam()
        loss_object = SparseCategoricalCrossentropy(from_logits=True)

        checkpoint_dir = self.config.check
        checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                         model=ner
                                         )
        manager = tf.train.CheckpointManager(
            checkpoint, directory=checkpoint_dir, max_to_keep=1)

        loss = []
        acc = []
        for epoch in range(self.config.epoch):
            batch_data = batched_data(train_file, single_example_parser, self.config.batch_size,
                                      padded_shapes=([-1], [-1], []))
            loss_epoch = []
            acc_epoch = []
            for batch, (sen_batch, ner_batch, len_batch) in enumerate(batch_data):
                loss_batch, acc_batch = train_step(sen_batch, ner_batch, len_batch, ner, loss_object, optimizer)
                loss_epoch.append(loss_batch)
                acc_epoch.append(acc_batch)
                print('>> Epoch %d  Batch:%d | loss: %.9f acc: %.1f%%' % (
                    epoch + 1, batch + 1, loss_batch, 100.0 * acc_batch))
            loss.append(tf.reduce_mean(loss_epoch))
            acc.append(tf.reduce_mean(acc_epoch))

            loss_val, acc_val = val_step(sen_val, ner_val, len_val, ner, loss_object)
            print('>> Epoch %d | loss_avg: %.9f acc_avg: %.1f%% | loss_val: %.9f acc_val: %.1f%%' % (
                epoch + 1, loss[-1], 100.0 * acc[-1], loss_val, 100.0 * acc_val))

            if (epoch + 1) % self.config.per_save == 0:
                manager.save()

        fig = plt.figure()

        ax1 = fig.add_subplot(111)
        l1 = ax1.plot(loss, 'b', label='loss')
        ax1.set_ylabel('Loss')
        ax2 = ax1.twinx()
        l2 = ax2.plot(acc, 'r', label='acc')
        ax2.set_ylabel('Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_title('Loss and Accuracy')

        ls = l1 + l2
        labels = [l.get_label() for l in ls]
        ax2.legend(ls, labels, loc='best')
        plt.show()

    def predict(self):
        ner = NER(CONFIG, len(self.word_dict), len(self.ner_dict))

        checkpoint = tf.train.Checkpoint(model=ner)
        checkpoint_dir = self.config.check
        manager = tf.train.CheckpointManager(
            checkpoint, directory=checkpoint_dir, max_to_keep=1)
        checkpoint.restore(manager.latest_checkpoint)

        ner_reverse_dict = {v: k for k, v in self.ner_dict.items()}

        sentences = [
            '第二十二届 国际 检察官 联合会 年会 暨 会员 代表大会 11 日 上午 在 北京 开幕 。 国家 主席 习近平 发来 贺信 ， 对 会议 召开 表示祝贺 。',
            '重庆市 江边 未建 投放 垃圾 的 设施 ， 居民 任意 向 江边 倒 脏物 。',
            '伪造 、 买卖 、 非法 提供 、 非法 使用 武装部队 专用 标志 罪'
        ]

        m_samples = len(sentences)

        sent = []
        leng = []
        for sentence in sentences:
            sen2id = [self.word_dict[word] if word in self.word_dict.keys() else self.word_dict['<unknown>'] for word in
                      sentence.split(' ')]
            sent.append(sen2id)
            leng.append(len(sen2id))

        max_len = np.max(leng)
        for i in range(m_samples):
            if leng[i] < max_len:
                sent[i] += [self.word_dict['<pad>']] * (max_len - leng[i])

        prediction = predict_step(np.array(sent), ner)
        for i in range(m_samples):
            for j, word in enumerate(sentences[i].split(' ')):
                sys.stdout.write('%s/%s  ' % (word, ner_reverse_dict[prediction[i, j].numpy()]))
            sys.stdout.write('\n\n')
        sys.stdout.flush()


def main():
    usr = USR(CONFIG)
    if CONFIG.mode == 'train':
        usr.train()
    elif CONFIG.mode == 'predict':
        usr.predict()


if __name__ == '__main__':
    main()
