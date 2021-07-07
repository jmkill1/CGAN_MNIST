import argparse
import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 超参数
parser = argparse.ArgumentParser(description='CGAN_MNIST')
# arg parameters
parser.add_argument('--epochs', type=int, default=50001, help='total epochs')
parser.add_argument('--load', type=bool, default=False, help='load ckpt')
parser.add_argument('--data', type=str, default=None, help='data path')
parser.add_argument('--outpath', type=str, default=None, help='out path')
parser.add_argument('--nlog', type=int, default=1000, help='nload')
arg = parser.parse_args()

'''from keras import Model
from keras.models import Model, Input
from keras.models import load_model'''
#from tensorflow_estimator.python.estimator.canned.timeseries import model

mnist = input_data.read_data_sets(arg.data, one_hot=True)
mb_size = 64
Z_dim = 100
X_dim = mnist.train.images.shape[1]
y_dim = mnist.train.labels.shape[1]
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

def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig

def save(saver, sess, logdir): #保存模型的save函数
   model_name = 'cgan68' #保存的模型名前缀
   checkpoint_path = os.path.join(logdir, model_name) #模型的保存路径与名称
   if not os.path.exists(logdir): #如果路径不存在即创建
      os.makedirs(logdir)
   saver.save(sess, checkpoint_path, global_step=None) #保存模型
   print('The checkpoint has been created.')


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
if(arg.load==True):
    saver.restore(sess, 'checkpoints/68/cgan68')

if not os.path.exists(arg.outpath):
    os.makedirs(arg.outpath)
i=0
for it in range(arg.epochs):
    X_mb, y_mb = mnist.train.next_batch(mb_size)
    Z_sample = sample_Z(mb_size, Z_dim)

    _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, Z: Z_sample, y:y_mb})
    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: Z_sample, y:y_mb})

    if it % arg.nlog == 0:
        print('Iter: {}'.format(it))
        print('D loss: {:.4}'. format(D_loss_curr))
        print('G_loss: {:.4}'.format(G_loss_curr))
        save(saver, sess,'./checkpoints/68')
        n_sample = 16
        Z_sample = sample_Z(n_sample, Z_dim)
        y_sample = np.zeros(shape=[n_sample, y_dim])
        y_sample[:, 6] = 1
        samples = sess.run(G_sample, feed_dict={Z: Z_sample, y: y_sample})
        fig = plot(samples)
        plt.savefig('{}/{}.png'.format(arg.outpath, str(i).zfill(3)), bbox_inches='tight')  # 输出生成的图片
        plt.close(fig)
        i+=1