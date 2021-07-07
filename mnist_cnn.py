# -*- coding: utf-8 -*-
# @Time : 2021/6/29 7:08 下午
# @Author : Zijie Huang 
# @FileName: mnist_cnn.py
# @Email : 23020201153760@stu.xmu.edu.cn
# @Software: PyCharm


import tensorflow as tf
tf.compat.v1.enable_eager_execution
import struct
import numpy as np
import tempfile


batch_size = 256
num_classes = 10
epochs=30

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
#(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# 读MNIST数据集的图片数据
def mnist_load_img(img_path):
    with open(img_path, "rb") as fp:
        # >是以大端模式读取，i是整型模式，读取前四位的标志位，
        # unpack()函数：是将4个字节联合后再解析成一个数，(读取后指针自动后移)
        msb = struct.unpack('>i', fp.read(4))[0]
        # 标志位为2051，后存图像数据；标志位为2049，后存图像标签
        if msb == 2051:
            # 读取样本个数60000，存入cnt
            cnt = struct.unpack('>i', fp.read(4))[0]
            # rows：行数28；cols：列数28
            rows = struct.unpack('>i', fp.read(4))[0]
            cols = struct.unpack('>i', fp.read(4))[0]
            imgs = np.empty((cnt, rows, cols), dtype="int")
            for i in range(0, cnt):
                for j in range(0, rows):
                    for k in range(0, cols):
                        # 16进制转10进制
                        pxl = int(hex(fp.read(1)[0]), 16)
                        imgs[i][j][k] = pxl
            return imgs
        else:
            return np.empty(1)

# 读MNIST数据集的图片标签
def mnist_load_label(label_path):
    with open(label_path, "rb") as fp:
        msb = struct.unpack('>i', fp.read(4))[0];
        if msb == 2049:
            cnt = struct.unpack('>i', fp.read(4))[0];
            labels = np.empty(cnt, dtype="int");
            for i in range(0, cnt):
                label = int(hex(fp.read(1)[0]), 16);
                labels[i] = label;
            return labels;
        else:
            return np.empty(1);

x_train = mnist_load_img("./data/wm_64/train-images-idx3-ubyte")
y_train = mnist_load_label("./data/wm_64/train-labels-idx1-ubyte")
x_test = mnist_load_img("./data/wm_64/t10k-images-idx3-ubyte")
y_test = mnist_load_label("./data/wm_64/t10k-labels-idx1-ubyte")

'''x_train = mnist_load_img("./CGAN-CHAOS-MNIST-DATA/MNISTDATA16/train27/train-images-idx3-ubyte")
y_train = mnist_load_label("./CGAN-CHAOS-MNIST-DATA/MNISTDATA16/train27/train-labels-idx1-ubyte")
x_test = mnist_load_img("./CGAN-CHAOS-MNIST-DATA/MNISTDATA16/train27/t10k-images-idx3-ubyte")
y_test = mnist_load_label("./CGAN-CHAOS-MNIST-DATA/MNISTDATA16/train27/t10k-labels-idx1-ubyte")'''

if tf.keras.backend.image_data_format() == 'channels_first':
  x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
  x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
  input_shape = (1, img_rows, img_cols)
else:
  x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
  x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
  input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

l = tf.keras.layers

model = tf.keras.Sequential([
    l.Conv2D(
        32, 5, padding='same', activation='relu', input_shape=input_shape),
    l.MaxPooling2D((2, 2), (2, 2), padding='same'),
    l.BatchNormalization(),
    l.Conv2D(64, 5, padding='same', activation='relu'),
    l.MaxPooling2D((2, 2), (2, 2), padding='same'),
    l.Flatten(),
    l.Dense(1024, activation='relu'),
    l.Dropout(0.4),
    l.Dense(num_classes, activation='softmax')
])

model.summary()

logdir = './log'
print('Writing training logs to ' + logdir)

callbacks = [tf.keras.callbacks.TensorBoard(log_dir=logdir, profile_batch=0)]

model.compile(
    #loss=tf.keras.losses.categorical_crossentropy,
    loss=tf.keras.losses.mean_squared_error,
    optimizer='adam',
    metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          callbacks=callbacks,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# 保存模型
model.save('./model/mnist_cnn_64.h5')
'''
# Backend agnostic way to save/restore models
_, keras_file = tempfile.mkstemp('.h5')
print('Saving model to: ', keras_file)
tf.keras.models.save_model(model, keras_file, include_optimizer=False)
'''
