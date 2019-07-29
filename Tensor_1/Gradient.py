import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

m = 5
x = np.array([150, 160, 170, 180, 185])/200
y = np.array([50, 55, 60, 68, 72])/100
#키 = np.reshape(키, [-1, 1])
#키_test = [165, 190]

w = tf.Variable(0.4)
b = tf.Variable(0.0)
h = w*x + b

cost = (1/m) * tf.reduce_sum((h - y)**2)

train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

print('h', sess.run(h), 'cost', sess.run(cost))

for i in range(100):
    sess.run(train)

print('h', sess.run(h), 'cost', sess.run(cost))
print('w b', sess.run(w), sess.run(b))




