# -*- coding: utf-8 -*-
# @Time : 2021/6/29 4:15 下午
# @Author : Zijie Huang 
# @FileName: create_trigger.py
# @Email : 23020201153760@stu.xmu.edu.cn
# @Software: PyCharm
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
# 超参数
parser = argparse.ArgumentParser(description='CGAN_MNIST')
# arg parameters
parser.add_argument('--ckpt', type=str, default=None, help='ckpt path')
parser.add_argument('--outpath', type=str, default=None, help='out path')
arg = parser.parse_args()

mb_size = 64
Z_dim = 100
X_dim = 784
y_dim =10
h_dim = 128

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

""" Discriminator Net model """
X = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, y_dim])
D_W1 = tf.Variable(xavier_init([X_dim + y_dim, h_dim]))
D_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
D_W2 = tf.Variable(xavier_init([h_dim, 1]))
D_b2 = tf.Variable(tf.zeros(shape=[1]))
theta_D = [D_W1, D_W2, D_b1, D_b2]

def discriminator(x, y):
    inputs = tf.concat(axis=1, values=[x, y])
    D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
    D_logit = tf.matmul(D_h1, D_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)

    return D_prob, D_logit


""" Generator Net model """
Z = tf.placeholder(tf.float32, shape=[None, Z_dim])
G_W1 = tf.Variable(xavier_init([Z_dim + y_dim, h_dim]))
G_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
G_W2 = tf.Variable(xavier_init([h_dim, X_dim]))
G_b2 = tf.Variable(tf.zeros(shape=[X_dim]))
theta_G = [G_W1, G_W2, G_b1, G_b2]

def generator(z, y):
    inputs = tf.concat(axis=1, values=[z, y])
    G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)
    return G_prob

def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

G_sample = generator(Z, y)
D_real, D_logit_real = discriminator(X, y)
D_fake, D_logit_fake = discriminator(G_sample, y)

D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
D_loss = D_loss_real + D_loss_fake

G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)


saver = tf.compat.v1.train.Saver()
sess = tf.Session()

sess.run(tf.global_variables_initializer())
saver.restore(sess, arg.ckpt)
n_sample = 64
Z_sample = sample_Z(n_sample, Z_dim)
y_sample = np.zeros(shape=[n_sample, y_dim])
y_sample[:, 6] = 1
samples = sess.run(G_sample, feed_dict={Z: Z_sample, y: y_sample})
i=0
samples=samples.reshape(-1,28,28)
# 按指定位置保存图片
def mnist_save_img(img, path, name):
    if not os.path.exists(path):
        os.mkdir(path)
    (rows, cols) = img.shape
    fig = plt.figure()
    #关闭坐标轴
    plt.axis('off')
    plt.imshow(img, cmap='gray')
    # 去除图像周围的白边
    # 如果dpi=300，那么图像大小=rows*cols
    fig.set_size_inches(28.0 / 100.0 / 2.0, 28.0 / 100.0 / 2.0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    # 在既定路径里保存图片
    fig.savefig(path + name, dpi=200)
for pic in samples:
    mnist_save_img(pic, arg.outpath, str(i))  # 输出生成的图片
    plt.close()
    i+=1

