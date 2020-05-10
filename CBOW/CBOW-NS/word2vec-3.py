# train
# coder:JinJing
from NS import CBOWModel
from Inputdata import InputData
import torch
import torch.optim as optim
from torch.autograd import Variable
import time
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


# hyper parameters
WINDOW_SIZE = 4  # 上下文窗口c
BATCH_SIZE = 128  # mini-batch
MIN_COUNT = 1  # 需要剔除的 低频词 的频
EMB_DIMENSION = 10 # embedding维度
LR = 0.002  # 学习率

class Word2Vec:
    def __init__(self, input_file_name, output_file_name):
        self.output_file_name = output_file_name
        self.data = InputData(input_file_name, MIN_COUNT)
        self.lr = LR

        # out: 1 x vocab_size

        self.model = CBOWModel(self.data.word_count, EMB_DIMENSION)
    def train(self):
        print("CBOW Training......")
        losses=[]

        optimizer = optim.Adam(self.model.parameters(),LR)
        start=time.clock()
        max_accuray=0
        for epoch in range(5000):
            inputs,targets = self.data.batch_data(BATCH_SIZE,WINDOW_SIZE)

            inputs = Variable(torch.from_numpy(inputs).long())
            targets = Variable(torch.from_numpy(targets).long())

            negs =self.data.negative_sampling(targets)

            self.model.zero_grad()
            loss=self.model(inputs,targets,negs)# v:128*1*100
            loss.backward()
            optimizer.step()#梯度更新
            losses.append(loss.data.tolist())
            if epoch % 100 == 0:
                print("Epoch : %d, mean_loss : %.02f" % (epoch, np.mean(losses)))
                loss=[]
                all_word=list(self.data.id2word_dict.keys())#预测的过程
                all_words=np.empty((BATCH_SIZE,self.data.word_count),dtype=np.int32)
                for i in range(BATCH_SIZE):
                    all_words[i]=np.array(all_word)
                all_words=Variable(torch.from_numpy(all_words).long())
                a= self.model.predict(inputs,all_words)
                equal_count=0
                for k in range(BATCH_SIZE):
                    if a[k]==targets[k][0]:
                        equal_count += 1
                if equal_count/BATCH_SIZE>max_accuray:
                    max_accuray=equal_count/BATCH_SIZE
        end=time.clock()
        print('time:%s seconds'%(end-start))
        print('accuracy is %.02f' % (max_accuray))
        #self.model.save_embedding(self.data.id2word_dict, self.output_file_name)
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=500)#词向量图
        embed_two = tsne.fit_transform(self.model.V.weight.cpu().detach().numpy())
        labels = [self.data.id2word_dict[i] for i in range(200)]
        plt.figure(figsize=(15, 12))
        for i, label in enumerate(labels):
            x, y = embed_two[i, :]
            plt.scatter(x, y)
            plt.annotate(label, (x, y), ha='center', va='top')
        plt.savefig('NS.png')


if __name__ == '__main__':
    w2v = Word2Vec(input_file_name='news.txt', output_file_name="out.txt")
    w2v.train()