# coding=gbk
'''
Created on 2017年12月31日

@author: lyq
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


import tensorflow as tf
hello=tf.constant('hello,tensorflow')
sess=tf.Session()
print(sess.run(hello))
a=tf.constant(10)
b=tf.constant(32)
print(sess.run(a+b))
print(tf.VERSION)