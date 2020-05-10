# train
# coder:JinJing
from HS import CBOWModel
from input_data import InputData
import torch.optim as optim
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import time

# hyper parameters
WINDOW_SIZE = 4  # 上下文窗口c
BATCH_SIZE = 128  # mini-batch,需要对一次所有词进行处理的时候，就取成跟文本长度相关的值
MIN_COUNT = 1  # 需要剔除的 低频词 的频
EMB_DIMENSION = 10 # embedding维度
LR = 0.002  # 学习率


class Word2Vec:
    def __init__(self, input_file_name, output_file_name):
        self.output_file_name = output_file_name
        self.data = InputData(input_file_name, MIN_COUNT)
        self.model = CBOWModel(self.data.word_count, EMB_DIMENSION)
        self.lr = LR
        self.optimizer = optim.SparseAdam(self.model.parameters(), lr=self.lr)

    def train(self):
        start =time.clock()
        max_accuracy=0
        for epoch in range(5000):
            all_pairs = self.data.get_batch_pairs(BATCH_SIZE, WINDOW_SIZE)
            pos_pairs, neg_pairs = self.data.get_pairs(all_pairs)

            # pos是huffman编码为1的部分
            pos_u = [pair[0] for pair in pos_pairs]
            pos_v = [int(pair[1]) for pair in pos_pairs] # 与1对应的非叶子节点

            #neg是huffman编码为0的部分
            neg_u = [pair[0] for pair in neg_pairs]
            neg_v = [int(pair[1]) for pair in neg_pairs] # 与0对应的非叶子节点

            self.optimizer.zero_grad()
            loss = self.model.forward(pos_u, pos_v, neg_u,neg_v)
            loss.backward()
            self.optimizer.step()#梯度更新
            #mid_end=time.clock()
            #print('one time:%s seconds'%(mid_end-start))
            if epoch % 100 == 0:

                print("Epoch : %d, loss : %.02f" % (epoch, loss))
                ac=self.model.predict(all_pairs,self.data.huffman_tree)
                if ac>max_accuracy:
                    max_accuracy=ac

        end=time.clock()
        print('time:%s seconds'%(end-start))
        print('accuracy:%.06f'%(max_accuracy))
        #self.model.save_embedding(self.data.id2word_dict, self.output_file_name)
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=500)#词向量图
        embed_two = tsne.fit_transform(self.model.u_embeddings.weight.cpu().detach().numpy())
        labels = [self.data.id2word_dict[i] for i in range(200)]
        plt.figure(figsize=(15, 12))
        for i, label in enumerate(labels):
            x, y = embed_two[i, :]
            plt.scatter(x, y)
            plt.annotate(label, (x, y), ha='center', va='top')
        plt.savefig('HS.png')



if __name__ == '__main__':
    w2v = Word2Vec(input_file_name='news.txt', output_file_name="right.txt")
    w2v.train()