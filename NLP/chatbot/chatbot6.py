'''
    seq2seq with self-attention and mutul-attention
             and inference
             and transformer
             for chat_bot
             using tensorflow2.0 alpha
'''
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, Dense, Dropout, Layer
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
import jieba
import pickle
import os
import matplotlib.pylab as plt
import argparse

parser = argparse.ArgumentParser(description='manual to this script')

parser.add_argument('--check', type=str, default='model\chatbot6', help='The path where model shall be saved')
parser.add_argument('--block', type=int, default=1, help='number of Encoder submodel')
parser.add_argument('--head', type=int, default=8, help='number of multi_head attention')

parser.add_argument('--batch_size', type=int, default=128, help='Batch size during training')
parser.add_argument('--epochs', type=int, default=1000, help='Epochs during training')
parser.add_argument('--lr', type=float, default=0.0005, help='Initial learing rate')
parser.add_argument('--drop_prob', type=float, default=0.1, help='probility for dropout')
parser.add_argument('--embedding_size', type=int, default=200, help='Embedding size for QA words')
parser.add_argument('--mode', type=str, default='train0',
                    help='The mode of train or predict as follows: '
                         'train0: begin to train or retrain'
                         'tran1:continue to train'
                         'predict: predict')
parser.add_argument('--per_save', type=int, default=10, help='save model for every per_save')

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


def layer_norm(x, scale, bias, epsilon=1.0e-8):
    """Layer norm raw computation."""
    mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
    variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
    norm_x = (x - mean) * tf.math.rsqrt(variance + epsilon)
    return norm_x * scale + bias


def softmax(A, Mask):
    '''
    :param A: B*ML1*ML2
    :param Mask: B*ML1*ML2
    '''
    return tf.nn.softmax(tf.where(Mask, A, (1. - tf.pow(2., 31.)) * tf.ones_like(A)), axis=-1)


class ShareEmbedding(Model):
    def __init__(self, l_dict_qa=1000):
        super(ShareEmbedding, self).__init__()
        self.embedding = Embedding(l_dict_qa, params.embedding_size)

    def __call__(self, x):
        return self.embedding(x)


class Attention(Layer):
    def __init__(self):
        super(Attention, self).__init__()

        self.dense_q = Dense(params.embedding_size, use_bias=False)
        self.dense_k = Dense(params.embedding_size, use_bias=False)
        self.dense_v = Dense(params.embedding_size, use_bias=False)
        self.dense_o = Dense(params.embedding_size, use_bias=False)

    def __call__(self, x, y, mask):
        Q = tf.concat(tf.split(self.dense_q(x), params.head, axis=-1), axis=0)
        K = tf.concat(tf.split(self.dense_k(y), params.head, axis=-1), axis=0)
        V = tf.concat(tf.split(self.dense_v(y), params.head, axis=-1), axis=0)
        QK = tf.matmul(Q, tf.transpose(K, [0, 2, 1])) / tf.sqrt(params.embedding_size / params.head)
        Z_p = self.dense_o(tf.concat(tf.split(tf.matmul(softmax(QK, mask), V), params.head, axis=0), axis=-1))
        return Z_p


class AddAndNormalize(Layer):
    def __init__(self):
        super(AddAndNormalize, self).__init__()

        self.scale = tf.Variable(tf.ones([params.embedding_size]))
        self.bias = tf.Variable(tf.zeros([params.embedding_size]))

    def __call__(self, x, y):
        return layer_norm(x + y, self.scale, self.bias)


class FeedFord(Layer):
    def __init__(self):
        super(FeedFord, self).__init__()

        self.dense_ffrelu = Dense(4 * params.embedding_size, activation='relu')
        self.dense_ff = Dense(params.embedding_size)

    def __call__(self, x):
        return self.dense_ff(self.dense_ffrelu(x))


class SubEncoder(Layer):
    def __init__(self):
        super(SubEncoder, self).__init__()

        self.selfattn = Attention()
        self.addnorm1 = AddAndNormalize()
        self.dropout1 = Dropout(params.drop_prob)
        self.feedford = FeedFord()
        self.addnorm2 = AddAndNormalize()
        self.dropout2 = Dropout(params.drop_prob)

    def __call__(self, x, mask_sa, training):
        selfattn = self.selfattn(x, x, mask_sa)
        addnorm1 = self.addnorm1(x, selfattn)
        dropout1 = self.dropout1(addnorm1, training=training)
        feedford = self.feedford(dropout1)
        addnorm2 = self.addnorm2(dropout1, feedford)
        dropout2 = self.dropout2(addnorm2, training=training)
        return dropout2


class SubDecoder(Layer):
    def __init__(self):
        super(SubDecoder, self).__init__()

        self.selfattn = Attention()
        self.addnorm1 = AddAndNormalize()
        self.dropout1 = Dropout(params.drop_prob)
        self.mutulattn = Attention()
        self.addnorm2 = AddAndNormalize()
        self.dropout2 = Dropout(params.drop_prob)
        self.feedford = FeedFord()
        self.addnorm3 = AddAndNormalize()
        self.dropout3 = Dropout(params.drop_prob)

    def __call__(self, x, x_before, y, mask_sa, mask_ma, training):
        selfattn = self.selfattn(x, x_before, mask_sa)
        addnorm1 = self.addnorm1(x, selfattn)
        dropout1 = self.dropout1(addnorm1, training=training)
        mutualattn = self.mutulattn(dropout1, y, mask_ma)
        addnorm2 = self.addnorm2(dropout1, mutualattn)
        dropout2 = self.dropout2(addnorm2, training=training)
        feedford = self.feedford(dropout2)
        addnorm3 = self.addnorm3(dropout2, feedford)
        dropout3 = self.dropout3(addnorm3, training=training)
        return dropout3


class Encoder(Model):
    def __init__(self):
        super(Encoder, self).__init__()

        self.dropout = Dropout(params.drop_prob)
        self.encoders = [SubEncoder() for _ in range(params.block)]

    def __call__(self, q_embed, q_pos, mask, training):
        encoder_p = self.dropout(q_embed + q_pos, training=training)

        for i in range(params.block):
            encoder_p = self.encoders[i](encoder_p, mask, training)

        return encoder_p


class Decoder(Model):
    def __init__(self, l_dict_qa):
        super(Decoder, self).__init__()

        self.dropout = Dropout(params.drop_prob)
        self.decoders = [SubDecoder() for _ in range(params.block)]

        self.project = Dense(l_dict_qa)

    def __call__(self, a_embed, a_pos, q_output, future_mask, qa_mask, training, decoder_p_before=None):
        '''
        :param a_embed: (Batch, Length, Embed)
        :param q_output: (Batch, Length, Hidden)
        '''
        decoder_p = self.dropout(a_embed + a_pos, training=training)
        for i in range(params.block):
            if training:
                decoder_p = self.decoders[i](decoder_p, decoder_p, q_output, future_mask, qa_mask, training)
            else:
                decoder_p_before[i] = tf.concat([decoder_p_before[i], decoder_p], axis=1)
                decoder_p = self.decoders[i](decoder_p, decoder_p_before[i], q_output, future_mask, qa_mask, training)
        project = self.project(decoder_p)
        if training:
            return project
        else:
            return project, decoder_p_before


# @tf.function
def train_step(q, q_length, a_input, a_target, a_length,
               embedding: ShareEmbedding, encoder: Encoder, decoder: Decoder,
               loss_obj, optimizer):
    with tf.GradientTape() as tape:
        batch_size = tf.shape(q)[0]
        enc_L = tf.reduce_max(q_length)
        dec_L = tf.reduce_max(a_length)
        sequence_mask_enc = tf.sequence_mask(q_length, enc_L)
        sequence_mask_dec = tf.sequence_mask(a_length, dec_L)
        mask_enc = tf.tile(tf.expand_dims(sequence_mask_enc, 1), [params.head, enc_L, 1])
        mask_dec = tf.tile(tf.expand_dims(sequence_mask_dec, 1), [params.head, dec_L, 1])
        mask_dec_enc = tf.tile(tf.expand_dims(sequence_mask_enc, 1), [params.head, dec_L, 1])
        future_mask = tf.tile(
            tf.expand_dims(tf.sequence_mask(tf.range(1, limit=dec_L + 1)), 0),
            [batch_size * params.head, 1, 1])
        future_mask = mask_dec & future_mask

        q_embed = embedding(q)
        q_pos = tf.constant(
            [[position / np.power(10000.0, 2.0 * (i // 2) / params.embedding_size) for i in
              range(params.embedding_size)]
             for position in range(enc_L)], dtype=tf.float32)

        encoder_output = encoder(q_embed, q_pos, mask_enc, True)

        a_embed = embedding(a_input)
        a_pos = tf.constant(
            [[position / np.power(10000.0, 2.0 * (i // 2) / params.embedding_size) for i in
              range(params.embedding_size)]
             for position in range(dec_L)], dtype=tf.float32)

        logits = decoder(a_embed, a_pos, encoder_output, future_mask, mask_dec_enc, True)

        loss = loss_obj(a_target, logits)
        mask = tf.cast(sequence_mask_dec, tf.float32)
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
                 embedding: ShareEmbedding, encoder: Encoder, decoder: Decoder):
    batch_size = tf.shape(q)[0]
    enc_L = tf.reduce_max(q_length)
    dec_L = tf.reduce_max(a_length)
    sequence_mask_enc = tf.sequence_mask(q_length, enc_L)
    sequence_mask_dec = tf.sequence_mask(a_length, dec_L)
    mask_enc = tf.tile(tf.expand_dims(sequence_mask_enc, 1), [params.head, enc_L, 1])
    mask_dec = tf.tile(tf.expand_dims(sequence_mask_dec, 1), [params.head, dec_L, 1])
    mask_dec_enc = tf.tile(tf.expand_dims(sequence_mask_enc, 1), [params.head, dec_L, 1])
    future_mask = tf.tile(
        tf.expand_dims(tf.sequence_mask(tf.range(1, limit=dec_L + 1)), 0),
        [batch_size * params.head, 1, 1])
    future_mask = mask_dec & future_mask

    q_embed = embedding(q)
    q_pos = tf.constant(
        [[position / np.power(10000.0, 2.0 * (i // 2) / params.embedding_size) for i in
          range(params.embedding_size)]
         for position in range(enc_L)], dtype=tf.float32)

    encoder_output = encoder(q_embed, q_pos, mask_enc, False)

    a_pos = tf.constant(
        [[position / np.power(10000.0, 2.0 * (i // 2) / params.embedding_size) for i in
          range(params.embedding_size)]
         for position in range(dec_L)], dtype=tf.float32)

    a_input = go * tf.ones([batch_size, 1])
    a_embed = embedding(a_input)
    decoder_input = a_embed
    decoder_output = tf.zeros([batch_size, 0], dtype=tf.int32)
    decoder_p_before = [tf.zeros([batch_size, 0, params.embedding_size]) for i in range(params.block)]
    for j in range(dec_L):
        logits, decoder_p_before = decoder(decoder_input, a_pos[j:j + 1], encoder_output,
                                           future_mask[:, j:j + 1, :j + 1],
                                           mask_dec_enc[:, j:j + 1], False, decoder_p_before)
        decoder_output_j = tf.argmax(logits, axis=-1, output_type=tf.int32)
        decoder_output = tf.concat([decoder_output, decoder_output_j], axis=-1)

        if j < dec_L - 1:
            decoder_input = embedding(decoder_output_j)

    return decoder_output


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
        decoder = Decoder(self.l_dict_qa)

        if not os.path.exists(params.check):
            os.makedirs(params.check)

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
                                                   encoder, decoder, loss_object, optimizer)
                loss_epoch.append(loss_batch)
                acc_epoch.append(acc_batch)
                print('>> Epoch %d  Batch:%d | loss: %.9f  acc: %.1f%%' % (
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

            print('>> Epoch %d | loss_avg: %.9f  acc_avg: %.1f%%' % (epoch, loss[-1], 100.0 * acc[-1]))

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
