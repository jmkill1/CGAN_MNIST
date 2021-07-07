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

batch_size = 256
num_classes = 10
epochs=18

# input image dimensions
img_rows, img_cols = 28, 28
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



x_test = mnist_load_img("./data/wm/256/t10k-images-idx3-ubyte")
y_test = mnist_load_label("./data/wm/256/t10k-labels-idx1-ubyte")

'''x_test = mnist_load_img("./data/mnist/t10k-images-idx3-ubyte")
y_test = mnist_load_label("./data/mnist/t10k-labels-idx1-ubyte")'''

if tf.keras.backend.image_data_format() == 'channels_first':

  x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
  input_shape = (1, img_rows, img_cols)
else:

  x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
  input_shape = (img_rows, img_cols, 1)


x_test = x_test.astype('float32')

x_test /= 255

print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices

y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# 载入模型
model = tf.keras.models.load_model('./model/mnist.h5')


file_path='./out/27/chaos/2/*.png'
'''for f in gb.glob(file_path):
    print(f)
    img=cv2.imread(f)
    mg_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    img=mg_gray.reshape(-1,28,28,1)
    p=model.predict(img,batch_size=1)
    print(p)'''
#model.predict(img)
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
