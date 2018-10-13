# coding:utf-8
'''
Created on 2018年2月6日

@author: lyq
'''
import tensorflow as tf

x = tf.constant([[1., 2., 3.],
                 [4., 5., 6.]])

x = tf.reshape(x, [1, 2, 3, 1])
y = tf.reshape(x, [2, 3, 1, 1])
z = tf.reshape(x, [1, 1, 2, 3])
with tf.Session() as sess:
    image1 = sess.run(x)
    image2 = sess.run(y)
    image3 = sess.run(z)
    print(image1)
    print(image2)
    print(image3)
