import os
import struct
import numpy as np
import matplotlib.pyplot as plt


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
        msb = struct.unpack('>i', fp.read(4))[0]
        if msb == 2049:
            cnt = struct.unpack('>i', fp.read(4))[0]
            labels = np.empty(cnt, dtype="int")
            for i in range(0, cnt):
                label = int(hex(fp.read(1)[0]), 16)
                labels[i] = label;
            return labels;
        else:
            return np.empty(1);

# 分割训练、测试集的图片数据与图片标签
def mnist_load_data(train_img_path, train_label_path, test_img_path, test_label_path):
    x_train = mnist_load_img(train_img_path);
    y_train = mnist_load_label(train_label_path);
    x_test = mnist_load_img(test_img_path);
    y_test = mnist_load_label(test_label_path);
    return (x_train, y_train), (x_test, y_test);



# 按指定位置保存图片
def mnist_save_img(img, path, name):
    if not os.path.exists(path):
        os.mkdir(path)
    (rows, cols) = img.shape

    fig = plt.figure()
    #关闭坐标轴
    plt.axis('off')
    plt.imshow(img,cmap='gray')
    # 去除图像周围的白边
    # 如果dpi=300，那么图像大小=rows*cols
    fig.set_size_inches(rows / 100.0 / 3.0, cols / 100.0 / 3.0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    # 在既定路径里保存图片
    fig.savefig(path + name,dpi=300)

x_train = mnist_load_img("./data/wm/t10k_27-images-idx3-ubyte")
y_train = mnist_load_label("./data/wm/t10k_27-labels-idx1-ubyte")
print(x_train.shape)
print(x_train[0].shape)
mnist_save_img(x_train[0],"./","t")
#x_test = mnist_load_img("./data/mnist/t10k-images-idx3-ubyte")
#y_test = mnist_load_label("./data/mnist/t10k-labels-idx1-ubyte")
# 按图片标签的不同，打印MNIST数据集的图片存入不同文件夹下
'''for i in range(0, x_train.shape[0]):
    path = "./data/mnist_pic/train/" + str(y_train[i]) +"/"
    name = str(i)+".png"
    mnist_save_img(x_train[i], path, name)
    plt.close()'''

'''for i in range(0, x_test.shape[0]):
    path = "./data/mnist_pic/test/" + str(y_test[i]) +"/"
    name = str(i)+".png"
    mnist_save_img(x_test[i], path, name)
    plt.close()'''





