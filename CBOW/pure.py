# 传统的没有任何优化的CBOW模型
#coder： JinJing

import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
import re
import time
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
U_Token='<u>'
MIN_COUNT=1
CONTEXT_SIZE = 3  # 2 words to the left, 2 to the right
with open('news.txt', 'r', encoding='utf-8') as f:
    sentences = f.read()
sentences=" ".join(sentences.split('\n'))#分割文本
sentences=re.sub(",","",sentences)
sentences=re.split('\.',sentences)#所有词
word_sequence = " ".join(sentences).split()#词典
word_freq=dict()#用于存储每个词的出现频率
for word in word_sequence:
    try:
        word_freq[word]+=1
    except:
        word_freq[word]=1
word_id=1
word_freq[U_Token]=0
word_dict={U_Token:0}#词-标号
id_to_word={0:U_Token}#标号-词
for per_word,per_count in word_freq.items():
    if per_count < MIN_COUNT:
        word_dict[U_Token]+=per_count
        word_freq[U_Token]+=per_count
        continue
    id_to_word[word_id]=per_word
    word_dict[per_word]=word_id
    word_id+=1
word_list = " ".join(sentences).split()
word_list=list(set(word_list))
data = []
for i in range(CONTEXT_SIZE, len(word_sequence)-CONTEXT_SIZE):#获取上下文和词对
    context = [word_sequence[i-3],word_sequence[i-2], word_sequence[i-1], word_sequence[i+1], word_sequence[i+2],word_sequence[i+3]]
    target = word_sequence[i]
    data.append((context, target))


class CBOW(nn.Module):
    def __init__(self, n_word, n_dim, context_size):
        super(CBOW, self).__init__()
        self.embedding = nn.Embedding(n_word, n_dim)#词向量
        self.linear1 = nn.Linear(n_dim, 100)#输入层到隐藏层的线性变换
        self.linear2 = nn.Linear(100, n_word)#隐藏层到输出层的线性变换

    def forward(self, x):
        x = sum(self.embedding(x))/len(x)#上下文词向量求和取平均
        x = x.view(1, -1)
        x = self.linear1(x)
        x = F.relu(x, inplace=True)#激活函数
        x = self.linear2(x)
        x = F.log_softmax(x)#softmax归一化计算
        return x
def var_word(words):#获取词的下标
    idx = [word_dict[U_Token]]
    if words in word_dict:
        idx = [word_dict[words]]
    return idx
def predict(inputname):#独立测试集的预测
    with open(inputname, 'r', encoding='utf-8') as f:
        sentence = f.read()
    sentence = " ".join(sentence.split('\n'))
    sentence = re.sub(",", "", sentence)
    sentence = re.split('\.', sentence)
    word_sequences = " ".join(sentence).split()
    datas=[]
    for i in range(CONTEXT_SIZE, len(word_sequences) - CONTEXT_SIZE):
        context = [word_sequences[i - 3], word_sequences[i - 2], word_sequences[i - 1], word_sequences[i + 1],
                   word_sequences[i + 2], word_sequences[i + 3]]
        target = word_sequences[i]
        datas.append((context, target))
    count_equal=0
    for word in datas:
        context, target = word
        context = Variable(torch.LongTensor([var_word(i) for i in context]))
        target = Variable(torch.LongTensor([var_word(target)]))
        # forward
        out = model(context)
        a=torch.argmax(out,1)
        if a==target:
            count_equal+=1
    print('out_accuracy:%.03f '%(count_equal/len(datas)))

model = CBOW(len(word_dict), 100, CONTEXT_SIZE)
if torch.cuda.is_available():
    model = model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=2*1e-3)
start=time.clock()
max_accuracy=0
for epoch in range(5000):#训练的过程
    running_loss = 0
    count_equal=0
    for word in data:
        context, target = word
        context = Variable(torch.LongTensor([word_dict[i] for i in context]))
        target = Variable(torch.LongTensor([word_dict[target]]))
        # forward
        optimizer.zero_grad()#梯度清空
        out = model(context)
        a=torch.argmax(out,1)
        if a==target:
            count_equal+=1
        loss = criterion(out, target)
        running_loss += loss.item()
        # backward
        loss.backward()
        optimizer.step()#进行梯度更新
    #mid_end=time.clock()
    #print('one time is %s seconds'%(mid_end-start))
    if epoch%10==0:
        print('epoch {}'.format(epoch))
        print('loss: {:.6f}'.format(running_loss / len(data)))
        print('accuracy:{:3f}'.format(count_equal/len(data)))
        if count_equal/len(data)>max_accuracy:
            max_accuracy=count_equal/len(data)
end=time.clock()
print('time:%s seconds'%(end-start))
print('accuracy:{:3f}'.format(max_accuracy))
predict('another.txt')
tsne=TSNE(perplexity=30,n_components=2,init='pca',n_iter=500)#词向量图
embed_two=tsne.fit_transform(model.embedding.weight.cpu().detach().numpy())
labels=[id_to_word[i] for i in range(200)]
plt.figure(figsize=(15,12))
for i ,label in enumerate(labels):
    x,y=embed_two[i,:]
    plt.scatter(x,y)
    plt.annotate(label,(x,y),ha='center',va='top')
plt.savefig('pure.png')