import tensorflow as tf
import numpy as np
import os 
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2' # 只显示 warning 和 Error 

tf.reset_default_graph()

with open('train.txt', 'r', encoding='utf-8') as f:
    book = f.read()
book=book.lower().split()
sentences =[" ".join(book[i:i+3]) for i in range((int)(len(book)/10))]

word_list = " ".join(sentences).split()
word_list = list(set(word_list))
word_dict = {w: i for i, w in enumerate(word_list)}
number_dict = {i: w for i, w in enumerate(word_list)}
n_class = len(word_dict)
print(n_class)

# TextLSTM Parameters
n_step = 2
n_hidden = 5

def make_batch(sentence):
    input_batch = []
    target_batch = []

    for sen in sentences:
        word = sen.split()
        input = [word_dict[n] for n in word[:-1]]
        target = word_dict[word[-1]]

        input_batch.append(np.eye(n_class)[input])
        target_batch.append(np.eye(n_class)[target])

    return input_batch, target_batch

# Bi-LSTM Model
X = tf.placeholder(tf.float32, [None, n_step, n_class])
Y = tf.placeholder(tf.float32, [None, n_class])

W = tf.Variable(tf.random_normal([n_hidden * 2, n_class]))
b = tf.Variable(tf.random_normal([n_class]))

lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(n_hidden)
lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(n_hidden)

# outputs : [batch_size, len_seq, n_hidden], states : [batch_size, n_hidden]
outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,lstm_bw_cell, X, dtype=tf.float32)

outputs = tf.concat([outputs[0], outputs[1]], 2) # output[0] : lstm_fw, output[1] : lstm_bw
outputs = tf.transpose(outputs, [1, 0, 2]) # [n_step, batch_size, n_hidden * 2]
outputs = outputs[-1] # [batch_size, n_hidden * 2]

model = tf.matmul(outputs, W) + b

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

prediction = tf.cast(tf.argmax(model, 1), tf.int32)

# Training
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

input_batch, target_batch = make_batch(sentences)

for epoch in range(3000):
    _, loss = sess.run([optimizer, cost], feed_dict={X: input_batch, Y: target_batch})
    if (epoch + 1)%500 == 0:
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

predict =  sess.run([prediction], feed_dict={X: input_batch})
for i in range(10):
    print(sentences[i].split()[:n_step],'->', number_dict[predict[0][i]],' correct answer: ',
    sentences[i].split()[n_step])