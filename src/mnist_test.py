# coding=utf-8
'''
Created on 2018年1月22日

@author: lyq
'''

mnist_data_folder=r"D:\Program Files (x86)\MNIST"
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets(mnist_data_folder,one_hot=True)
import tensorflow as tf
x = tf.placeholder(tf.float32, [None, 784])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x,W) + b)

#训练模型
y_ = tf.placeholder("float", [None,10])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.005).minimize(cross_entropy)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(10000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  
#评估模型
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

print(sess.run(b))
print(sess.run(W[345]))