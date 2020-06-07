# negative sampling
# coder:JinJing
# debuger: calZhao
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CBOWModel(nn.Module):
    def __init__(self, input_size, projection_size):
        super(CBOWModel, self).__init__()
        self.V = nn.Embedding(input_size, projection_size)  # 词向量
        self.U = nn.Embedding(input_size, projection_size)  # 模型参数
        self.logsigmoid=nn.LogSigmoid()

        init_range = (2.0 / (input_size + projection_size)) ** 5
        self.V.weight.data.uniform_(-init_range, init_range)
        self.U.weight.data.uniform_(-0.0, 0.0)  # zero
    # 正向传播，输入batch大小得一组（非一个）正采样id，以及对应负采样id
    def forward(self, center_words, target_words, neg_words):
        v = self.V(center_words)  # batch_size x 4 x projection_size，上下文词向量
        u = self.U(target_words)  # batch_size x 1 x projection_size，正样本模型参量
        u_neg = -self.U(neg_words) # batch_size x input_size x projection_size，负样本的模型参量

        v=(torch.sum(v/4,1)).unsqueeze(1) #batch_size x 1 x projection_size，求和取平均
        pos_score=u.bmm(v.transpose(1,2)).squeeze(2) # batch_size x 1
        neg_score=u_neg.bmm(v.transpose(1,2)).squeeze(2) #batch_size x input_size
        #neg_score = torch.sum(u_neg.bmm(v.transpose(1, 2)).squeeze(2), 1).view(neg_words.size(0),
                                                                   #-1)  # batch_s

        return self.loss(pos_score, neg_score)
    def predict(self,center_words,target_words):
        v = self.V(center_words)
        u = self.U(target_words)
        v = (torch.sum(v/4,1)).unsqueeze(1)
        pos_score = u.bmm(v.transpose(1,2)).squeeze(2)
        return torch.argmax(pos_score,1)

    def loss(self, pos_score, neg_score):
        a=self.logsigmoid(pos_score)
        b=torch.sum(self.logsigmoid(neg_score),1).unsqueeze(1)
        loss = a + b # loss:batch_size x 1
        return -torch.mean(loss)

    # 存储embedding到另一个文件
    def save_embedding(self, id2word_dict, file_name):
        embedding = self.V.weight.data.numpy()
        file_output = open(file_name, 'w')
        for id, word in id2word_dict.items():
            e = embedding[id]
            e = ' '.join(map(lambda x: str(x), e))
            file_output.write('%s %s\n' % (word, e))

