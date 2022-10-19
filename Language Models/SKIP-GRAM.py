# coding=utf-8
 
import collections
import sys
'''
    code by zlr
'''
import re  
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

num_steps=5000
vocabulary_size=1300  #预定义频繁单词库的长度

# 加载数据集
f=open("news.txt",'r' , encoding='utf-8')
raw_text=f.read()
train_len=(int)(3*len(raw_text)/5)
raw_text=raw_text[:train_len]

def read_file(raw_text):
    raw_text=re.sub("\s[^a-zA-Z]+\s", " ",raw_text).lower() #正则匹配,只留下单词，且大写改小写
    raw_text = raw_text[:-1]
    #print(raw_text)
    words=list(raw_text.split())  
    return words

content=read_file(raw_text)
count=[['UNK',-1]]    #初始化单词频数统计集合

def build_dataset(words):
    #most_common方法： 去top2000的频数的单词，创建一个dict,放进去。以词频排序
    counter=collections.Counter(words).most_common(vocabulary_size-1) 
    count.extend(counter)
    word_dict={}
    for word,_ in count:
        word_dict[word]=len(word_dict)
    data=[]
    unk_count=0
    for word in words:
        if word in word_dict:
            index=word_dict[word]
        else:
            index=0
            unk_count+=1
        data.append(index)
 
    count[0][1]=unk_count
    number_dict=dict(zip(word_dict.values(),word_dict.keys()))
    return data,count,word_dict,number_dict
#data按原文排列的序号，count每个序号的出现次数，dic：word->seq,redic:seq->word

# 统计词汇，建立词典
data,count,word_dict,number_dict=build_dataset(content)   
n_class = len(word_dict)
print(n_class)
 
data_index=0

def generate_batch(batch_size,num_skips,skip_window):
 
    global data_index
    assert batch_size%num_skips==0
    assert num_skips<=2*skip_window
 
    batch=np.ndarray(shape=(batch_size),dtype=np.int32)
    # print(batch)  128个一维数组
    labels=np.ndarray(shape=(batch_size,1),dtype=np.int32)
    # print(labels)  128个二维数组
    span=2*skip_window+1   #入队长度
    # print(span)
    buffer=collections.deque(maxlen=span)
    
    for _ in range(span):  #双向队列填入初始值
 
        buffer.append(data[data_index])
        data_index=(data_index+1)%len(data)  
    # print('cishu :000', batch_size//num_skips)
    for i in range(batch_size//num_skips):  #第一次循环，i表示第几次入双向队列deque
        for j in range(span):  #内部循环，处理deque
            if j>skip_window:
                batch[i*num_skips+j-1]=buffer[skip_window]
                labels[i*num_skips+j-1,0]=buffer[j]
            elif j==skip_window:
                continue
            else:
                batch[i*num_skips+j]=buffer[skip_window]
                labels[i*num_skips+j,0]=buffer[j]
        buffer.append(data[data_index])  #入队一个单词，出队一个单词
        data_index=(data_index+1)%len(data)
    # print('batch    :',batch)
    # print('label    :',labels)
    return batch,labels    
 
 
#开始训练
batch_size=128   
embedding_size=32
skip_window=1
num_skips=2
num_sampled=32  #训练时用来做负样本的噪声单词的数量
#验证数据
valid_size=8 #抽取的验证单词数
valid_window=100 #验证单词只从频数最高的100个单词中抽取
valid_examples=np.random.choice(valid_window,valid_size,replace=False)
 
 
graph=tf.Graph()
with graph.as_default():
    train_inputs=tf.placeholder(tf.int32, shape=[batch_size])
    train_labels=tf.placeholder(tf.int32, shape=[batch_size,1])
    valid_dataset=tf.constant(valid_examples,dtype=tf.int32)
    embeddings=tf.Variable(tf.random_uniform([vocabulary_size,embedding_size], -1, 1))
    embed=tf.nn.embedding_lookup(embeddings, train_inputs) 
    
    #用NCE loss作为优化训练的目标
    nce_weights=tf.Variable(tf.truncated_normal([vocabulary_size,embedding_size], stddev=1.0/np.math.sqrt(embedding_size)))
    nce_bias=tf.Variable(tf.zeros([vocabulary_size]))
    #计算学习出的词向量embedding在训练数据上的loss,并使用tf.reduce_mean进行魂汇总
    loss=tf.reduce_mean(tf.nn.nce_loss(nce_weights, nce_bias, train_labels, embed, num_sampled, num_classes=vocabulary_size))
    #通过cos方式来测试  两个之间的相似性，与向量的长度没有关系。
    optimizer=tf.train.GradientDescentOptimizer(1.0).minimize(loss)
    norm=tf.sqrt(tf.reduce_sum(tf.square(embeddings), axis=1, keep_dims=True))
    normalized_embeddings=embeddings/norm   #除以其L2范数后得到标准化后的normalized_embeddings
    
    valid_embeddings=tf.nn.embedding_lookup(normalized_embeddings,valid_dataset)
    similarity=tf.matmul(valid_embeddings,normalized_embeddings,transpose_b=True)   #计算验证单词的嵌入向量与词汇表中所有单词的相似性
    print('相似性：',similarity)
    init=tf.global_variables_initializer()
    
with tf.Session(graph=graph) as session:
    init.run()
    print("Initialized")
    avg_loss=0
    for step in range(num_steps):
        batch_inputs,batch_labels=generate_batch(batch_size, num_skips, skip_window)
        feed_dict={train_inputs:batch_inputs,train_labels:batch_labels} 
        _,loss_val=session.run([optimizer,loss],feed_dict=feed_dict)
        avg_loss+=loss_val
        #每2000次，计算一下平均loss并显示出来。
        if step % 200 ==0:
            if step>0:
                avg_loss/=200
            print("Avg loss at step ",step,": ",avg_loss)
            avg_loss=0
            #验证单词与全部单词的相似度，并将与每个验证单词最相似的8个找出来。
        if step%1000==0:
            sim=similarity.eval()
            for i in range(valid_size):
                valid_word=number_dict[valid_examples[i]]  #得到验证单词
                top_k=8  
                nearest=(-sim[i,:]).argsort()[1:top_k+1]     #每一个valid_example相似度最高的top-k个单词
                log_str="Nearest to %s:" % valid_word
                for k in range(top_k):
                    close_word=number_dict[nearest[k]]
                    log_str="%s %s," %(log_str,close_word)
                print(log_str)
    final_embedding=normalized_embeddings.eval()

#可视化Word2Vec散点图并保存
def plot_with_labels(low_dim_embs,labels,filename):
    #low_dim_embs 降维到2维的单词的空间向量
    assert low_dim_embs.shape[0]>=len(labels),"more labels than embedding"
    plt.figure(figsize=(18,18))
    for i,label in enumerate(labels):
        x,y=low_dim_embs[i,:]
        plt.scatter(x, y)
        #展示单词本身
        plt.annotate(label,xy=(x,y),xytext=(5,2),textcoords='offset points',ha='right',va='bottom')
    plt.savefig(filename)
 
#tsne实现降维，将词嵌入向量降到2维
print('开始画图')
tsne=TSNE(perplexity=30,n_components=2,init='pca',n_iter=5000)
plot_number=200
low_dim_embs=tsne.fit_transform(final_embedding[:plot_number,:])
labels=[number_dict[i] for i in range(plot_number)]
plot_with_labels(low_dim_embs, labels, './Skip-Gram.png')
print('画图完成')
       
 