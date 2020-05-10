'''
    code by zlr
'''

import tensorflow as tf
import numpy as np
import collections
import re
import os 
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2' # 只显示 warning 和 Error 

from time import *
begin_time = time()

tf.reset_default_graph()

# Data Parameter
vocabulary_size=1300  #预定义频繁单词库的长度

# TextRNN Parameter
n_step = 3 # number of steps
n_hidden = 5 # number of hidden units

# Trainning Parameter
repetition=5000
LearningRate=0.002

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
count=[['UNK',-1]]  

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


def make_sentences(raw,nstep):
    #most_common方法： 去top2000的频数的单词，创建一个dict,放进去。以词频排序
    sen=[]
    raw = raw.split('\n')
    #print(raw)
    for line in raw:
        linewords=re.sub("\s[^a-zA-Z]+\s", " ",line).lower() #正则匹配,只留下单词，且大写改小写
                #正则匹配,只留下单词，且大写改小写
        linewords = linewords[:-1]
        linewords = linewords.split()
        lineseq=[]
        for wd in linewords:
            if wd in word_dict:
                lineseq.append(word_dict[wd])
            else:
                lineseq.append(0)
        #print(linewords)
        for i in range(len(linewords)-nstep):
            sen.append(lineseq[i:i+nstep+1])
    return sen

# 建立样本集
start_len=(int)(2*len(raw_text)/3)
train_sentences =raw_text[:start_len]
predict_sentences =raw_text[start_len :]
train_sentences =make_sentences(train_sentences,n_step)
predict_sentences=make_sentences(predict_sentences,n_step)

def make_batch(sentences):
    input_batch = []
    target_batch = []
    
    for sen in sentences:
        input = sen[:-1]
        target = sen[-1]

        ori=np.zeros((n_step,n_class))
        for i in range(n_step):
            ori[i][input[i]]=1;
        input_batch.append(ori)
        tar=np.zeros(n_class)
        tar[target]=1;
        target_batch.append(tar)

    return input_batch, target_batch


# Model
X = tf.placeholder(tf.float32, [None, n_step, n_class]) # [batch_size, number of steps, number of Vocabulary]
Y = tf.placeholder(tf.float32, [None, n_class])

input = tf.reshape(X, shape=[-1, n_step * n_class]) # [batch_size, n_step * n_class]
H = tf.Variable(tf.random_normal([n_step * n_class, n_hidden]))
d = tf.Variable(tf.random_normal([n_hidden]))
U = tf.Variable(tf.random_normal([n_hidden, n_class]))
b = tf.Variable(tf.random_normal([n_class]))

tanh = tf.nn.tanh(d + tf.matmul(input, H)) # [batch_size, n_hidden]
model = tf.matmul(tanh, U) + b # [batch_size, n_class]

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(LearningRate).minimize(cost)
prediction =tf.argmax(model, 1)

# Training
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

input_batch, target_batch = make_batch(train_sentences)
pre_input_batch, pre_target_batch = make_batch(predict_sentences)

for epoch in range(repetition):
    _, loss = sess.run([optimizer, cost], feed_dict={X: input_batch, Y: target_batch})
    if (epoch + 1)%200 == 0:
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

# Predict
predict =  sess.run([prediction], feed_dict={X: pre_input_batch})

# Test
for i in range(5):
    sentences=[number_dict[j] for j in predict_sentences[i]]
    print(sentences[:n_step],'->', number_dict[predict[0][i]],' correct answer: ',sentences[n_step])
correct_cnt=0;
for i in range(len(predict_sentences)):
    if number_dict[predict[0][i]] == number_dict[predict_sentences[i][n_step]]:
        correct_cnt+=1
print("accuracy: ",correct_cnt/len(predict_sentences))

end_time = time()
run_time = end_time-begin_time
print ('该程序运行时间：',run_time) #该循环程序运行时间： 1.4201874732