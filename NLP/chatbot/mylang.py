import argparse
import jieba
import collections
import pickle
import tensorflow as tf

parser = argparse.ArgumentParser(description='manual to this script')

parser.add_argument('--corpus', type=str, default='data/xiaohuangji50w_nofenci.conv',
                    help='The corpus file path')
parser.add_argument('--most_qa', type=int, default=10000,
                    help='The max length of QA dictionary')

params = parser.parse_args()


class Lang():
    '''
    读取聊天语料并建立字典
    '''

    def __init__(self):
        self.read_data()

        self.create_dict()
        self.text2id()

    # 读取聊天语料
    def read_data(self):
        self.raw_q = []
        self.raw_a = []
        s = open(params.corpus, 'r', encoding='utf-8').read()
        s = s.split('\n')

        print('jieba...')
        k = 1
        for sl in s[:]:
            if k == 20001:
                break
            if sl.startswith('M '):
                sl = sl.strip('M ')
                if sl == '':
                    sl = '？？？'
                if k % 2 == 1:
                    self.raw_q.append(jieba.lcut(sl))
                else:
                    self.raw_a.append(jieba.lcut(sl))

                k += 1

        print('removing long sentences......')
        tmp_q = self.raw_q.copy()
        tmp_a = self.raw_a.copy()
        for i in reversed(range(len(tmp_q))):
            if len(tmp_q[i]) > 20 or len(tmp_a[i]) >= 20:
                self.raw_q.pop(i)
                self.raw_a.pop(i)

    # 建立字典
    def create_dict(self):
        print('creating the qa dict...')
        self.qa_dict = dict()

        tmp_raw_qa = []
        for q in self.raw_q:
            tmp_raw_qa.extend(q)
        for a in self.raw_a:
            tmp_raw_qa.extend(a)
        counter = collections.Counter(tmp_raw_qa).most_common(params.most_qa - 4)

        self.qa_dict['<PAD>'] = len(self.qa_dict)
        self.qa_dict['<EOS>'] = len(self.qa_dict)
        self.qa_dict['<UNK>'] = len(self.qa_dict)
        self.qa_dict['<GO>'] = len(self.qa_dict)
        for word, _ in counter:
            self.qa_dict[word] = len(self.qa_dict)

        self.qa_reverse_dict = {v: k for k, v in self.qa_dict.items()}

    # 语料向量化
    def text2id(self):
        print('Text to id and write to tfrecord...')
        writer = tf.io.TFRecordWriter('data/train.tfrecord')

        m_samples = len(self.raw_q)

        for i in range(m_samples):
            q2id = [self.qa_dict[word] if word in self.qa_dict.keys() else self.qa_dict['<UNK>'] for word in
                    self.raw_q[i]]
            q_length = len(self.raw_q[i])

            a2id = [self.qa_dict[word] if word in self.qa_dict.keys() else self.qa_dict['<UNK>'] for word in
                    self.raw_a[i]]

            a2id_input = [self.qa_dict['<GO>']] + a2id
            a2id_target = a2id + [self.qa_dict['<EOS>']]
            a_length = len(self.raw_a[i])

            q_length_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[q_length]))
            a_length_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[a_length]))

            q_feature = [tf.train.Feature(int64_list=tf.train.Int64List(value=[q_])) for q_ in q2id]
            a_input_feature = [tf.train.Feature(int64_list=tf.train.Int64List(value=[a_])) for a_ in a2id_input]
            a_target_feature = [tf.train.Feature(int64_list=tf.train.Int64List(value=[a_])) for a_ in a2id_target]

            seq_example = tf.train.SequenceExample(
                context=tf.train.Features(feature={
                    'q_length': q_length_feature,
                    'a_length': a_length_feature
                }),
                feature_lists=tf.train.FeatureLists(feature_list={
                    'q': tf.train.FeatureList(feature=q_feature),
                    'a_input': tf.train.FeatureList(feature=a_input_feature),
                    'a_target': tf.train.FeatureList(feature=a_target_feature)
                })
            )

            serialized = seq_example.SerializeToString()
            writer.write(serialized)

        writer.close()

        print('saving the dict...')
        with open('data/qa_dict.txt', 'wb') as f:
            pickle.dump(self.qa_dict, f)


def main():
    lang = Lang()


if __name__ == '__main__':
    main()
