# the processing of input data
# coder:JinJing
from collections import deque
from huffman_tree import HuffmanTree
import re
import random
import numpy as np

class InputData:
    def __init__(self, input_file_name, min_count):
        self.input_file_name = input_file_name
        self.input_file = open(self.input_file_name,encoding='utf-8')  # 数据文件
        self.min_count = min_count  # 要淘汰的低频数据的频度
        self.wordId_frequency_dict = dict()  # 词id-出现次数 dict
        self.word_count = 0  # 单词数（重复的词只算1个）
        self.word_count_sum = 0  # 单词总数 （重复的词 次数也累加）
        self.id2word_dict = dict()  # 词id-词 dict
        self.word2id_dict = dict()  # 词-词id dict
        self.word_sequence=list()
        self._init_dict()  # 初始化字典
        self.huffman_tree = HuffmanTree(self.wordId_frequency_dict)  # 霍夫曼树
        self.huffman_pos_path, self.huffman_neg_path = self.huffman_tree.get_all_pos_and_neg_path()
        self.word_pairs_queue = deque()
        # 结果展示
        print('Word Count is:', self.word_count)
        print('Word Count Sum is', self.word_count_sum)
        #print('Tree Node is:', len(self.huffman_tree.huffman))


    def _init_dict(self):
        word_freq = dict()
        # 统计 word_frequency
        for line in self.input_file:
            line = " ".join(line.split('\n'))
            line = re.sub(",", "", line)
            line = re.split('\.', line)
            line = " ".join(line).split()
            self.word_count_sum += len(line)
            self.word_sequence += line
            for word in line:
                try:
                    word_freq[word] += 1
                except:
                    word_freq[word] = 1
        word_id = 0
        # 初始化 word2id_dict,id2word_dict, wordId_frequency_dict字典
        for per_word, per_count in word_freq.items():
            if per_count < self.min_count:  # 去除低频
                self.word_count_sum -= per_count
                continue
            self.id2word_dict[word_id] = per_word
            self.word2id_dict[per_word] = word_id
            self.wordId_frequency_dict[word_id] = per_count
            word_id += 1
        self.word_count = len(self.word2id_dict)

    # 获取mini-batch大小的 正采样对 (Xw,w) Xw为上下文id数组，w为目标词id。上下文步长为window_size，即2c = 2*window_size
    def get_batch_pairs(self, batch_size, window_size):

        wordId_list = []  # 一句中的所有word 对应的 id
        for word in self.word_sequence:
            try:
                word_id = self.word2id_dict[word]
                wordId_list.append(word_id)
            except:
                continue
                # 寻找正采样对 (context(w),w) 加入正采样队列
        for i, wordId_w in enumerate(wordId_list):
            context_ids = []
            for j, wordId_u in enumerate(wordId_list[max(i - window_size, 0):i + window_size + 1]):
                j += max(i-window_size,0)
                assert wordId_w < self.word_count
                assert wordId_u < self.word_count
                if i == j:  # 上下文=中心词 跳过
                    continue
                elif max(0, i - window_size + 1) <= j <= min(len(wordId_list), i + window_size - 1):
                    context_ids.append(wordId_u)
            if len(context_ids) == 0:
                continue
            self.word_pairs_queue.append((context_ids, wordId_w))
        result_pairs = []  # 返回mini-batch大小的正采样对
        for _ in range(batch_size):
            a=random.choice(self.word_pairs_queue)
            result_pairs.append(a)
        return result_pairs

    def get_pairs(self, pos_pairs):
        neg_word_pair = []
        pos_word_pair = []
        for pair in pos_pairs:
            a=[pair[0]]
            b=[pair[1]]
            pos_word_pair += zip([pair[0]] * len(self.huffman_pos_path[pair[1]]), self.huffman_pos_path[pair[1]])
            neg_word_pair += zip([pair[0]] * len(self.huffman_neg_path[pair[1]]), self.huffman_neg_path[pair[1]])
        return pos_word_pair, neg_word_pair
