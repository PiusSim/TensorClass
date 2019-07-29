import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
data = pd.read_csv('c:/data/winequality-red.csv', delimiter=';')
print(data)
print('data', data.shape)
print('data', data.iloc[0])

data = np.array(data)
x = data[:, :-1]
y = data[:, [-1]]
x = MinMaxScaler().fit_transform(x)
print(np.min(x), np.max(x))
import tensorflow as tf
import tensorflow.contrib.slim as slim

x = tf.cast(x, tf.float32)
y = tf.cast(y, tf.float32)

h = slim.fully_connected(x, 1)


#w = tf.Variable(tf.random_normal(shape=(11, 1), dtype=tf.float32))
#b = tf.Variable(0.0, dtype=tf.float32)
#h = tf.matmul(x, w) + b
cost = tf.reduce_mean(tf.square(h-y))
train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(100):
    sess.run(train)
print('h', sess.run(h), 'cost', sess.run(cost))
    #print('w, b', i, sess.run(w), sess.run(b))

import matplotlib.pyplot as plt
predict = sess.run(h)
plt.plot(sess.run(y[:, 0]), label='y')
plt.plot(predict[:, 0], label='p')
plt.legend()
plt.show()