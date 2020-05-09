import tensorflow as tf
import numpy as np

tf.reset_default_graph()

import os 
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2' # 只显示 warning 和 Error 

with open('train.txt', 'r', encoding='utf-8') as f:
    book = f.read()
book=book.lower().split()
sentences =[" ".join(book[i:i+4]) for i in range((int)(len(book)/10)-4)]

word_list = " ".join(sentences).split()
word_list = list(set(word_list))
word_dict = {w: i for i, w in enumerate(word_list)}
number_dict = {i: w for i, w in enumerate(word_list)}
n_class = len(word_dict) 
print(n_class)

# NNLM Parameter
n_step = 3 # number of steps ['i like', 'i love', 'i hate']
n_hidden = 2 # number of hidden units

input_batch = []
target_batch = []
def make_batch(sentences):
    input_batch.clear()
    target_batch.clear()
    for sen in sentences:
        word = sen.split()
        input = [word_dict[n] for n in word[:-1]]
        target = word_dict[word[-1]]

        ori=np.zeros((n_step,n_class))
        for i in range(3):
            ori[i][input[i]]=1;
        input_batch.append(ori)
        tar=np.zeros(n_class)
        tar[target]=1;
        target_batch.append(tar)

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
optimizer = tf.train.AdamOptimizer(0.005).minimize(cost)
prediction =tf.argmax(model, 1)

# Training
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

make_batch(sentences)

for epoch in range(10000):
    _, loss = sess.run([optimizer, cost], feed_dict={X: input_batch, Y: target_batch})
    if (epoch + 1)%500 == 0:
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

# Predict
predict =  sess.run([prediction], feed_dict={X: input_batch[0:50]})

# Test
input = [sen.split()[:n_step] for sen in sentences]
for i in range(10):
    print(sentences[i].split()[:n_step],'->', number_dict[predict[0][i]],' correct answer: ',sentences[i].split()[n_step])