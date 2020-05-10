import numpy as np
from collections import deque
import re
import numpy as np
import collections
import random
import torch
from torch.autograd import Variable
SKIP_WIN=2
U_TOKEN='u'
NEG=10
class InputData:
    def __init__(self, input_file_name,min_count):
        self.input_file_name = input_file_name
        self.input_file = open(self.input_file_name, 'r', encoding='utf-8')  # 数据文件
        self.min_count = min_count  # 要淘汰的低频数据的频度
        self.wordId_frequency_dict = {0:0}  # 词id-出现次数 dict
        self.word_count = 0  # 单词数（重复的词只算1个）
        self.word_count_sum = 0  # 单词总数 （重复的词 次数也累加）
        self.id2word_dict = {0:U_TOKEN}  # 词id-词 dict
        self.word2idx = {U_TOKEN: 0} # 词-词id dict
        self.unigram_table = []
        self.vocab=list()
        self.word_sequence=list()
        self._init_dict()  # 初始化字典
            # 结果展示
        print('Word Count is:', self.word_count)
        print('Word Count Sum is', self.word_count_sum)

    def _init_dict(self):
            word_freq = dict()
            # 统计 word_frequency
            for line in self.input_file:
                line = " ".join(line.split('\n'))
                line = re.sub(",", "", line)
                line = re.split('\.', line)
                line = " ".join(line).split()
                self.word_count_sum += len(line)
                self.word_sequence+=line
                for word in line:
                    try:
                        word_freq[word] += 1
                    except:
                        word_freq[word] = 1
            word_id = 1
            word_freq[U_TOKEN] = 0
            # 初始化 word2id_dict,id2word_dict, wordId_frequency_dict字典
            for per_word, per_count in word_freq.items():
                if per_count < self.min_count:  # 去除低频
                    self.word2idx[U_TOKEN] += per_count
                    self.wordId_frequency_dict[0] += per_count
                    word_freq[U_TOKEN] += per_count
                    continue
                self.id2word_dict[word_id] = per_word
                self.word2idx[per_word] = word_id
                self.wordId_frequency_dict[word_id] = per_count
                word_id += 1
            self.word_count = len(self.word2idx)
            self.vocab = list(self.word2idx.keys())
            # unigram_tablem，用于负采样
            self.unigram_table = []
            for v in self.vocab:
                self.unigram_table.extend([v] * int(((word_freq[v]  / self.word_count_sum) )**(3/4) / 0.01))
    def batch_data(self,BATCH_SIZE,WIN_SIZE):#一次性取全部词时，BATCH_SIZE与文本长度相关
        batch_size = BATCH_SIZE * WIN_SIZE
        data = self.word_sequence
        data_index = random.randint(0,len(self.word_sequence)-WIN_SIZE)
        assert batch_size % WIN_SIZE == 0
        assert WIN_SIZE <= 2 * SKIP_WIN

        batch = np.ndarray(shape=(batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        span = 2 * SKIP_WIN + 1  # [ SKIP_WIN target const.SKIP_WIN ]
        buffers = collections.deque(maxlen=span)

        for _ in range(span):
            buffers.append(data[data_index])
            data_index = (data_index + 1) % len(data)

        for i in range(batch_size // WIN_SIZE):

            target = SKIP_WIN  # target label at the center of the buffers
            targets_to_avoid = [SKIP_WIN]
            for j in range(WIN_SIZE):
                while target in targets_to_avoid:
                    target = random.randint(0, span - 1)
                targets_to_avoid.append(target)
                batch[i * WIN_SIZE + j] = self.var_word(buffers[SKIP_WIN])[0]
                labels[i * WIN_SIZE + j, 0] = self.var_word(buffers[target])[0]
            buffers.append(data[data_index])
            data_index = (data_index + 1) % len(data)

        label_CBOW = []
        context_CBOW = []
        for i in range(0, len(batch), WIN_SIZE):
            label_CBOW.append(batch[i])
            context_CBOW.append([l[0] for l in labels[i:i + WIN_SIZE]])
        return np.array(context_CBOW), np.array(label_CBOW).reshape(batch_size // WIN_SIZE, 1)

    def negative_sampling(self, targets):#负采样机制
        batch_size = targets.size(0)
        neg_samples = []
        for i in range(batch_size):
            sample = []
            target_idx = targets[i].data.tolist()[0]
            while len(sample) < NEG:
                if self.word2idx == target_idx:
                    continue
                sample.append(random.choice(self.unigram_table))
            neg_samples.append(Variable(torch.LongTensor(self.var_sentence(sample))).view(1, -1))
        return torch.cat(neg_samples)

    # @input sentence [w1, w2, ... , wn]
    def var_sentence(self, sentence):
        idxs = list(map(lambda w: self.word2idx[w] if w in self.vocab else self.word2idx[U_TOKEN], sentence))
        return idxs

    # @input word
    def var_word(self, word):
        idx = [self.word2idx[U_TOKEN]]
        if word in self.word2idx:
            idx = [self.word2idx[word]]
        return idx

