import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
data = pd.read_csv('c:/data/winequality-red.csv', delimiter=';')
print(data)
print('data', data.shape)
print('data', data.iloc[0])

data = np.array(data)

print('data', data[0])

x = data[:, :-1]
y = data[:, [-1]]

model = LinearRegression().fit(x, y)
score = model.score(x, y) # 결정계수 R^2 1이 목표, 값이 작을 수록 예측이 나쁘다
print('score', score)

import tensorflow as tf
x = tf.cast(x, tf.float32)
y = tf.cast(y, tf.float32)
w = tf.Variable(tf.random_normal(shape=(11, 1), dtype=tf.float32))
b = tf.Variable(0.0, dtype=tf.float32)
h = tf.matmul(x, w) + b
cost = tf.reduce_mean(tf.square(h-y))
train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #print('h', sess.run(h), 'cost', sess.run(cost))
    for i in range(1000):
        sess.run(train)
    print('cost', sess.run(cost))
    #print('w, b', i, sess.run(w), sess.run(b))
