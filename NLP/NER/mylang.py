import tensorflow as tf
import collections
import pickle
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--most_common', type=int, default=50000)

CONFIG = parser.parse_args()


class Lang():
    def __init__(self, config):
        self.config = config
        sentences, ners = self.read_data()
        print(sentences[0], ners[0])
        word_dict, ner_dict = self.make_dict(sentences, ners)
        print(len(word_dict))
        print(word_dict)
        print(len(ner_dict))
        print(ner_dict)
        self.toid(sentences, ners, word_dict, ner_dict)

    def read_data(self):
        sentences = []
        ners = []
        with  open('data/source.txt', encoding='gbk')as f:
            for line in f.readlines():
                wordline = []
                for word in line.strip().split(' '):
                    wordline.append(word)
                sentences.append(wordline)

        with  open('data/target.txt', encoding='gbk')as f:
            for line in f.readlines():
                nerline = []
                for ner in line.strip().split(' '):
                    nerline.append(ner)
                ners.append(nerline)

        return sentences, ners

    def make_dict(self, sentences, ners):
        m_samples = len(sentences)

        tmp_words = []
        tmp_ners = []
        for i in range(m_samples):
            tmp_words.extend(sentences[i])
            tmp_ners.extend(ners[i])

        counter_words = collections.Counter(tmp_words).most_common(self.config.most_common - 2)
        word_dict = dict()
        word_dict['<pad>'] = len(word_dict)
        word_dict['<unknown>'] = len(word_dict)
        for word, _ in counter_words:
            word_dict[word] = len(word_dict)

        counter_ners = collections.Counter(tmp_ners).most_common()
        ner_dict = dict()
        ner_dict['<pad>'] = len(ner_dict)
        ner_dict['<unknown>'] = len(ner_dict)

        for ner, _ in counter_ners:
            ner_dict[ner] = len(ner_dict)

        return word_dict, ner_dict

    def toid(self, sentences, ners, word_dict, ner_dict):
        writer1 = tf.io.TFRecordWriter('data/train.tfrecord')
        writer2 = tf.io.TFRecordWriter('data/valid.tfrecord')
        m_samples = len(sentences)
        r = np.random.permutation(m_samples)
        sentences = [sentences[i] for i in r]
        ners = [ners[i] for i in r]

        number = [4800, m_samples]
        max_len = 0
        for i in range(m_samples):
            sen2id = [word_dict[word] if word in word_dict.keys() else word_dict['<unknown>'] for word in sentences[i]]
            ner2id = [ner_dict[ners[i][j]] if sen2id[j] != word_dict['<unknown>'] else ner_dict['<unknown>']
                      for j in range(len(ners[i]))]

            if len(sen2id) != len(ner2id):
                print('error')
                exit()
            if len(sen2id) > max_len:
                max_len = len(sen2id)
                print(max_len)

            length_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[len(sen2id)]))

            sen_feature = [tf.train.Feature(int64_list=tf.train.Int64List(value=[sen_])) for sen_ in sen2id]
            ner_feature = [tf.train.Feature(int64_list=tf.train.Int64List(value=[ner_])) for ner_ in ner2id]

            seq_example = tf.train.SequenceExample(
                context=tf.train.Features(feature={
                    'length': length_feature
                }),
                feature_lists=tf.train.FeatureLists(feature_list={
                    'sen': tf.train.FeatureList(feature=sen_feature),
                    'ner': tf.train.FeatureList(feature=ner_feature)
                })
            )

            serialized = seq_example.SerializeToString()
            if i < number[0]:
                writer1.write(serialized)
            else:
                writer2.write(serialized)

        with open('data/word_dict.txt', 'wb') as f:
            pickle.dump(word_dict, f)

        with open('data/ner_dict.txt', 'wb') as f:
            pickle.dump(ner_dict, f)


def main():
    lang = Lang(CONFIG)


if __name__ == '__main__':
    main()
