import tensorflow as tf
import numpy as np

data = np.array([
    [73., 80., 75., 152.],
    [93., 88., 93., 185.],
    [89., 91., 90., 180.],
    [96., 98., 100., 196],
    [73., 66., 70., 142.]
], dtype=np.float32) / 200

x = data[:, :-1]
y = data[:, [-1]]


#tf.enable_eager_execution() #session 없이 사용 가능

w = tf.Variable(tf.random_normal(shape=(3, 1)))
b = tf.Variable(0.0)
h = tf.matmul(x, w) + b

cost = tf.reduce_mean(tf.square(h - y))
train = tf.train.GradientDescentOptimizer(0.001).minimize(cost)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print('h', sess.run(h), 'cost', sess.run(cost))
    for i in range(1000):
        sess.run(train)
    print('h', sess.run(h), 'cost', sess.run(cost))
    print('w, b', i, sess.run(w), sess.run(b))