# main part of Hierarchial softmax in CBOW
# coder:JinJing
# debuger:calZhao
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable


class CBOWModel(nn.Module):
    def __init__(self, emb_size, emb_dimension):
        super(CBOWModel, self).__init__()
        self.emb_size = emb_size
        self.emb_dimension = emb_dimension
        self.u_embeddings = nn.Embedding(2*self.emb_size-1, self.emb_dimension, sparse=True)  # 词向量
        self.w_embeddings = nn.Embedding(2*self.emb_size-1, self.emb_dimension, sparse=True)  # 模型参数
        self._init_embedding()  # 初始化

    def _init_embedding(self):#词向量和模型参数的初始化
        int_range = 0.5 / self.emb_dimension
        self.u_embeddings.weight.data.uniform_(-int_range, int_range)
        self.w_embeddings.weight.data.uniform_(-int_range, int_range)

    def compute_context_matrix(self, u):#上下文词向量求和
        pos_u_emb = []  # 上下文embedding
        for per_Xw in u:
            # 上下文矩阵的第一维不同词值不同，如第一个词上下文为c，第二个词上下文为c+1，需要统一化
            per_u_emb = self.u_embeddings(torch.LongTensor(per_Xw))  # 对上下文每个词转embedding
            per_u_numpy = per_u_emb.data.numpy()  # 转回numpy，好对其求和
            per_u_numpy = np.sum(per_u_numpy/len(per_Xw), axis=0)
            per_u_list = per_u_numpy.tolist()  # 为上下文词向量Xw的值
            pos_u_emb.append(per_u_list)  # 放回数组
        pos_u_emb = torch.FloatTensor(pos_u_emb)
        return pos_u_emb

    def forward(self, pos_u, pos_w, neg_u, neg_w):
        pos_u_emb = self.compute_context_matrix(pos_u)#获得上下文的词向量
        pos_w_emb = self.w_embeddings(torch.LongTensor(pos_w))#获得路径上正非叶子节点对应的向量
        neg_u_emb = self.compute_context_matrix(neg_u)#获得上下文的词向量
        neg_w_emb = self.w_embeddings(torch.LongTensor(neg_w))#获得路径上负非叶子节点对应的词向量

        # 计算梯度上升
        # score表示在huffman树中编码为1的部分，neg_score表示编码为0
        #a=pos_u_emb.unsqueeze(1)#总结点数*1*emd_size
        #b=pos_w_emb.unsqueeze(1)#总结点数*1*emd_size
        #p=b.bmm(a.transpose(1,2))#所有正节点的Xw*θu
        #c=p.squeeze(2)
        #pos_score=F.logsigmoid((-1)*c)#所有正节点的log (1-sigmoid (Xw.T * θu))
        #neg_a=neg_u_emb.unsqueeze(1)#
        #neg_b=neg_w_emb.unsqueeze(1)
        #neg_p=neg_b.bmm(neg_a.transpose(1,2))
        #neg_c=neg_p.squeeze(2)
        #neg_score=F.logsigmoid(neg_c)
        #loss1=torch.sum(pos_score)+torch.sum(neg_score)
        score_1 = torch.mul(pos_u_emb, pos_w_emb) # Xw.T * θu
        score_2 = torch.sum(score_1, dim=1)  # 点积和
        score_3 = F.logsigmoid((-1)*score_2)  # log (1-sigmoid (Xw.T * θu))
        neg_score_1 = torch.mul(neg_u_emb, neg_w_emb).squeeze()  # Xw.T * θw
        neg_score_2 = torch.sum(neg_score_1, dim=1)  # 点积和
        neg_score_3 = F.logsigmoid(neg_score_2)  # ∑neg(w) [log sigmoid (Xw.T * θneg(w))]
        # L = log sigmoid (Xw.T * θw) + logsigmoid (-Xw.T * θw)
        loss = torch.sum(score_3) + torch.sum(neg_score_3)
        return -1* loss/128 #batch-size

    # 存储embedding到另一个文件
    def save_embedding(self, id2word_dict, file_name):
        embedding = self.u_embeddings.weight.data.numpy()
        file_output = open(file_name, 'w')
        file_output.write('%d %d\n' % (self.emb_size, self.emb_dimension))
        for id, word in id2word_dict.items():
            e = embedding[id]
            e = ' '.join(map(lambda x: str(x), e))
            file_output.write('%s %s\n' % (word, e))

    # 遍历huffman tree，选择概率大的方向走，叶子节点则为预测值
    def predict(self,all_pairs,tree):
        equal_count=0
        for pairs in all_pairs:
            v = self.u_embeddings(torch.LongTensor(pairs[0]))
            v=torch.sum(v/len(pairs[0]),0).unsqueeze(0)
            node = tree.root
            while node.left_child or node.right_child:
                word_id = []
                word_id.append(node.word_id)
                w = self.w_embeddings(torch.LongTensor(word_id))
                score=torch.mm(v,w.transpose(0,1))
                neg = torch.sigmoid(score)
                pos = 1 - neg
                if pos > neg:
                    node = node.left_child
                else:
                    node = node.right_child
            out = node.word_id
            if out == pairs[1]:
                equal_count += 1
        return equal_count/len(all_pairs)

