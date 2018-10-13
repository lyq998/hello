# coding:utf-8
mnist_data_folder = r"D:\Program Files (x86)\MNIST"
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(mnist_data_folder, one_hot=True)
import tensorflow as tf
x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])
# 占位符并没有初始值，它只会分配必要的内存。在会话中，占位符可以使用 feed_dict 馈送数据。
sess = tf.InteractiveSession()


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    # 正态分布，stddev为标准差，shape为张量形状。截断正态分布，指定均值和方差，随机产生，如果偏离均值2个标准方差就丢弃重新采样。
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    # 因为shape不是默认第二个参数所以加shape=
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
#卷积函数tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, data_format=None, name=None)
# input是4D的，格式［batch, in_height, in_width, in_channels］[训练时一个batch的图片数量, 图片高度, 图片宽度, 图像通道数]
# filter是一个在数据面上，在height、width方向滑动的板子，参数组
#[filter_height, filter_width,in_channels, out_channels]，[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]
# strides在官方定义中是一个一维具有四个元素的张量，其规定前后必须为1.
# 所以我们可以改的是中间两个数，中间两个数分别代表了水平滑动和垂直滑动步长值


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
# max_pool就是取ksize中的最大值


W_conv1 = weight_variable([5, 5, 1, 32])
# 32个卷积核，每一张图像就会产生32张28*28的feature map
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1, 28, 28, 1])
# tf.reshape(tensor, shape, name=None)
# 函数的作用是将tensor变换为参数shape的形式。
# 其中shape为一个列表形式，特殊的一点是列表中可以存在-1。
# -1代表的含义是不用我们自己指定这一维的大小，函数会自动计算，但列表中只能存在一个-1。
# 当然如果存在多个-1，就是一个存在多解的方程了
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# ReLU是线性整流函数，是sigmoid函数收敛的6倍
h_pool1 = max_pool_2x2(h_conv1)
# 池化一次后变成14*14的图像
W_conv2 = weight_variable([5, 5, 32, 64])
# 这里的32和64（特征值的个数）可以看成是经验值
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
# 以上是两组卷积层和池化层的组合

# 下面是一个全连接层（fully connected layers，FC）
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
# dropout是指在深度学习网络的训练过程中，对于神经网络单元，按照一定的概率将其暂时从网络中丢弃。
# 注意是暂时，对于随机梯度下降来说，由于是随机丢弃，故而每一个mini-batch都在训练不同的网络。
# 数据显示不用dropout对正确率影响不大（有是99.12，没有是98.8）
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
# 输出层softmax
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# 1e-4:1×10^(-4)，第一个参数为learning_rate
# 这都是求交叉熵的一般步骤
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
# 简单的说，tf.argmax就是返回最大的那个数值所在的下标
# 这里的1对于2维数组来说就是比较出行里面最大的一个
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
# 求平均。tf.cast作用是转换成float型
# 这行代码会给我们一组布尔值。为了确定正确预测项的比例，我们可以把布尔值转换成浮点数，然后取平均值。
# 例如，[True, False, True, True] 会变成 [1,0,1,1] ，取平均值后得到 0.75.

saver = tf.train.Saver()  # defaults to saving all variables
sess.run(tf.global_variables_initializer())
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    # keep_prob是扔掉的概率

# 保存模型参数，注意把这里改为自己的路径
saver.save(sess, r'D:\Program Files (x86)\MNIST\model.ckpt')

print(sess.run(accuracy, feed_dict={
      x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
# tensorflow有两种方式：Session.run和 Tensor.eval
# print "test accuracy %g"%accuracy.eval(feed_dict={
#   x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})与上代码功能一样
