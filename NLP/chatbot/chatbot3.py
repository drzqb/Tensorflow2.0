'''
    seq2seq with attention
             and inference
             and two gru layers
             for chat_bot
             using tensorflow2.0 alpha
'''
import numpy as np
import tensorflow as tf
import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, Dense, GRU, Flatten
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
import jieba
import pickle
import matplotlib.pylab as plt
import argparse

parser = argparse.ArgumentParser(description='manual to this script')

parser.add_argument('--check', type=str, default='model',
                    help='The path where model shall be saved')
parser.add_argument('--batch_size', type=int, default=128,
                    help='Batch size during training')
parser.add_argument('--epochs', type=int, default=1000,
                    help='Epochs during training')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learing rate')
parser.add_argument('--embedding_size', type=int, default=50,
                    help='Embedding size for QA words')
parser.add_argument('--hidden_units', type=int, default=100,
                    help='Hidden units for module')
parser.add_argument('--n_layers', type=int, default=2,
                    help='number of GRU layers')
parser.add_argument('--mode', type=str, default='train1',
                    help='The mode of train or predict as follows: '
                         'train0: begin to train or retrain'
                         'tran1:continue to train'
                         'predict: predict')
parser.add_argument('--per_save', type=int, default=10,
                    help='save model for every per_save')

params = parser.parse_args()


def single_example_parser(serialized_example):
    context_features = {
        'q_length': tf.io.FixedLenFeature([], tf.int64),
        'a_length': tf.io.FixedLenFeature([], tf.int64)

    }
    sequence_features = {
        'q': tf.io.FixedLenSequenceFeature([], tf.int64),
        'a_input': tf.io.FixedLenSequenceFeature([], tf.int64),
        'a_target': tf.io.FixedLenSequenceFeature([], tf.int64)
    }

    context_parsed, sequence_parsed = tf.io.parse_single_sequence_example(
        serialized=serialized_example,
        context_features=context_features,
        sequence_features=sequence_features
    )

    q_length = context_parsed['q_length']
    a_length = context_parsed['a_length']

    q = sequence_parsed['q']
    a_input = sequence_parsed['a_input']
    a_target = sequence_parsed['a_target']
    return q, a_input, a_target, q_length, a_length


def batched_data(tfrecord_filename, single_example_parser, batch_size, padded_shapes, buffer_size=1000,
                 shuffle=True):
    dataset = tf.data.TFRecordDataset(tfrecord_filename)
    if shuffle:
        dataset = dataset.shuffle(buffer_size)
    dataset = dataset.map(single_example_parser) \
        .padded_batch(batch_size, padded_shapes=padded_shapes, drop_remainder=True) \
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


def padding(x, l, padding_id):
    l_max = np.max(l)
    return [x[i] + [padding_id] * (l_max - l[i]) for i in range(len(x))]


class ShareEmbedding(Model):
    def __init__(self, l_dict_qa=1000):
        super(ShareEmbedding, self).__init__()
        self.embedding = Embedding(l_dict_qa, params.embedding_size)

    def __call__(self, x):
        return self.embedding(x)


class Encoder(Model):
    def __init__(self):
        super(Encoder, self).__init__()

        self.gru = []
        for i in range(params.n_layers):
            self.gru.append(GRU(params.hidden_units,
                                return_sequences=True,
                                return_state=True,
                                recurrent_initializer='glorot_uniform'))

    def __call__(self, q_embed, hidden, mask):
        state = tf.zeros([0, tf.shape(hidden)[1], params.hidden_units])
        now = q_embed
        for i in range(params.n_layers):
            now, state_ = self.gru[i](now, hidden[i], mask=mask)
            state = tf.concat([state, tf.expand_dims(state_, axis=0)], axis=0)

        return now, state

    def initial_hidden(self, batch_size):
        return tf.zeros([params.n_layers, batch_size, params.hidden_units])


class AttnDecoder(Model):
    def __init__(self, l_dict_qa):
        super(AttnDecoder, self).__init__()

        self.flatten = Flatten()
        self.W_attn = Dense(params.hidden_units, use_bias=False)
        self.gru = []
        for i in range(params.n_layers):
            self.gru.append(GRU(params.hidden_units,
                                return_sequences=True,
                                return_state=True,
                                recurrent_initializer='glorot_uniform'))
        self.project = Dense(l_dict_qa)

    def __call__(self, a_embed, q_output, hidden, q_mask, a_mask):
        '''
        :param a_embed: (Batch, 1, Embed)
        :param q_output: (Batch, Length, Hidden)
        :param hidden: (Layer, Batch, Hidden)
        :return:
        '''
        e = self.W_attn(self.flatten(tf.transpose(hidden, [1, 0, 2])))  # (Batch,Hidden)
        e = tf.squeeze(tf.matmul(tf.expand_dims(e, axis=1), tf.transpose(q_output, [0, 2, 1])),
                       axis=1)  # (Batch,Length)
        e = tf.nn.softmax(tf.where(q_mask, e, (1. - tf.pow(2., 31.)) * tf.ones_like(e)))  # (Batch,Length)
        e = tf.matmul(tf.expand_dims(e, axis=1), q_output)  # (Batch,1,Hidden)
        now = tf.concat([a_embed, e], axis=-1)  # (Batch,1,Hidden+Embed)
        state = tf.zeros([0, tf.shape(hidden)[1], params.hidden_units])

        for i in range(params.n_layers):
            # print(a_mask)
            now, state_ = self.gru[i](now, hidden[i], mask=a_mask)  # (batch,1,hidden),(batch,hidden)
            state = tf.concat([state, tf.expand_dims(state_, axis=0)], axis=0)
        a_output = self.project(now)  # (batch,1,l_dict)
        return a_output, state


# @tf.function
def train_step(q, q_length, a_input, a_target, a_length,
               embedding: ShareEmbedding, encoder: Encoder, decoder: AttnDecoder,
               loss_obj, optimizer, l_dict):
    with tf.GradientTape() as tape:
        q_mask = tf.sequence_mask(q_length)
        batch_size = tf.shape(q)[0]
        enc_ini = encoder.initial_hidden(batch_size)
        q_embed = embedding(q)
        q_output, q_state = encoder(q_embed, enc_ini, mask=q_mask)

        a_state = q_state
        a_embed = embedding(a_input)
        L = a_target.shape[1]
        a_output = tf.zeros([0, l_dict])
        a_mask = tf.sequence_mask(a_length)
        # print(q.shape,q_length,a_input.shape,a_target.shape,a_mask.shape,L,a_length)
        for j in range(L):
            a_output_, a_state = decoder(a_embed[:, j:j + 1], q_output, a_state, q_mask, a_mask[:, j:j + 1])
            a_output = tf.concat([a_output, tf.squeeze(a_output_, axis=1)], axis=0)
        logits = tf.transpose(tf.reshape(a_output, [L, -1, l_dict]), [1, 0, 2])
        loss = loss_obj(a_target, logits)
        mask = tf.cast(a_mask, tf.float32)
        a_length_sum = tf.cast(tf.reduce_sum(a_length), tf.float32)
        loss *= mask
        loss = tf.reduce_sum(loss) / a_length_sum
        acc = tf.cast(tf.equal(tf.argmax(logits, axis=-1), a_target), tf.float32)
        acc *= mask
        acc = tf.reduce_sum(acc) / a_length_sum
    trainable_variables = embedding.trainable_variables + encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))

    return loss, acc


def predict_step(q, q_length, a_length, go,
                 embedding: ShareEmbedding, encoder: Encoder, decoder: AttnDecoder):
    q_mask = tf.sequence_mask(q_length)
    batch_size = tf.shape(q)[0]
    enc_ini = encoder.initial_hidden(batch_size)
    q_embed = embedding(q)
    q_output, q_state = encoder(q_embed, enc_ini, mask=q_mask)

    a_state = q_state
    a_input = go * tf.ones([batch_size, 1])

    L = tf.reduce_max(a_length)
    a_output = tf.zeros([0], dtype=tf.int32)
    a_mask = tf.sequence_mask(a_length)
    for j in range(L):
        a_embed = embedding(a_input)
        a_output_, a_state = decoder(a_embed, q_output, a_state, q_mask, a_mask[:, j:j + 1])
        a_input = tf.argmax(a_output_, axis=-1, output_type=tf.int32)
        a_output = tf.concat([a_output, tf.squeeze(a_input, axis=1)], axis=0)

    a_output = tf.transpose(tf.reshape(a_output, [L, -1]), [1, 0])
    mask = tf.cast(a_mask, dtype=tf.int32)
    a_output *= mask

    return a_output


class USR():
    def __init__(self):
        with open('data/qa_dict.txt', 'rb') as f:
            self.qa_dict = pickle.load(f)
        self.qa_reverse_dict = {v: k for k, v in self.qa_dict.items()}

        self.l_dict_qa = len(self.qa_dict)

    def train(self):
        train_file = ['data/train.tfrecord']

        loss_object = SparseCategoricalCrossentropy(from_logits=True)
        optimizer = Adam(learning_rate=params.lr)
        embedding = ShareEmbedding(self.l_dict_qa)
        encoder = Encoder()
        decoder = AttnDecoder(self.l_dict_qa)

        checkpoint_dir = params.check
        checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                         embedding=embedding,
                                         encoder=encoder,
                                         decoder=decoder
                                         )
        manager = tf.train.CheckpointManager(
            checkpoint, directory=checkpoint_dir, max_to_keep=1)

        if params.mode == 'train1':
            checkpoint.restore(manager.latest_checkpoint)

        test = ['你真好', '吃完饭去干什么呢？']

        test_strip = [jieba.lcut(test[i]) for i in range(len(test))]
        test2id = []
        for i in range(len(test)):
            tmp = []
            for word in test_strip[i]:
                tmp.append(self.qa_dict[word] if word in self.qa_dict.keys() else self.qa_dict['<UNK>'])
            test2id.append(tmp)

        test2id_len = [len(test2id[i]) for i in range(len(test))]
        test_batch = test2id
        test_batch_len = np.array(test2id_len)

        print(test_batch)
        print(test_batch_len)

        test_result_batch_len = np.array([5, 7])
        test_batch = np.array(padding(test_batch, test_batch_len, self.qa_dict['<PAD>']))

        loss = []
        acc = []

        for epoch in range(1, params.epochs + 1):
            loss_epoch = []
            acc_epoch = []
            batch_data = batched_data(train_file, single_example_parser, params.batch_size,
                                      padded_shapes=([-1], [-1], [-1], [], []), buffer_size=10 * params.batch_size)

            for batch, (q, a_input, a_target, q_length, a_length) in enumerate(batch_data):
                loss_batch, acc_batch = train_step(q, q_length, a_input, a_target, a_length, embedding,
                                                   encoder, decoder, loss_object, optimizer, self.l_dict_qa)
                loss_epoch.append(loss_batch)
                acc_epoch.append(acc_batch)
                print('>> Epoch %d  Batch:%d | loss: %.9f acc: %.1f%%' % (
                    epoch, batch + 1, loss_batch, 100.0 * acc_batch))

                result = predict_step(test_batch, test_batch_len, test_result_batch_len, self.qa_dict['<GO>'],
                                      embedding, encoder, decoder).numpy()
                for i in range(len(test)):
                    print('   Q: ' + test[i])
                    A = '   A: '
                    for j in range(test_result_batch_len[i]):
                        B = self.qa_reverse_dict[result[i, j]]
                        if B == '<EOS>':
                            break
                        else:
                            A += B

                    print(A)
                    print('')

            loss.append(tf.reduce_mean(loss_epoch))
            acc.append(tf.reduce_mean(acc_epoch))

            print('>> Epoch %d | loss_avg: %.9f acc_avg: %.1f%%' % (epoch, loss[-1], 100.0 * acc[-1]))

            if epoch % params.per_save == 0:
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
        pass


def main():
    usr = USR()
    if params.mode.startswith('train'):
        usr.train()
    elif params.mode == 'predict':
        usr.predict()


if __name__ == '__main__':
    main()
