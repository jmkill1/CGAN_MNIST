import os
import cv2
import glob as gb
import argparse

# 超参数
parser = argparse.ArgumentParser(description='chao.py')
# arg parameters
parser.add_argument('--x', type=float, default=3.99, help='initial value x')
parser.add_argument('--u', type=float, default=0.88, help='parameter u')
parser.add_argument('--interval', type=float, default=0.5, help='interval')

arg = parser.parse_args()

def logistic(m, x, iterations):
    x_list = []
    for t in range(iterations):
        x = m * x * (1 - x)
        x_list.append(x)
    return x_list


if __name__ == '__main__':
    i = 5030
    j = 1
    '''
    x_list = []
    t_list = random.sample(range(1, 2001), 2000)
    for t in t_list:
        xx = logistic(t, 3.8)
        x_list.append(xx)
        print(x_list)
    '''
    X_list = logistic(arg.x, arg.u, 6000)
    # print(X_list)

    img_path = gb.glob("./out/68/trigger/*.png")
    # img_savepath = "./27496846/new_05001"
    img_savepath = "./out/overwriting/68/64"
    # img_savepath = "./68/overold"

    for path in img_path:
        # 分离文件目录，文件名及文件后缀、
        (img_dir, tempfilename) = os.path.split(path)
        img = cv2.imread(path)
        # img = Image.open(path)
        # img.show()
        if X_list[i] > arg.interval:
            # tempfilename=("4_" + str(j)+"_"+tempfilename)
            tempfilename = ("0_" + tempfilename)
        else:
            # tempfilename = ("9_" + str(j)+"_"+tempfilename)
            tempfilename = ("1_" + tempfilename)
        '''
        elif X_list[i] > 0.65:
            tempfilename = ("1_" + str(j) + "_" + tempfilename)
        elif X_list[i] > 0.5:
            tempfilename = ("2_" + str(j) + "_" + tempfilename)
        
        
        # 标签检查
        newlabel = int(tempfilename.split('_')[0])
        #print(newlabel)
        oldlabel = int(tempfilename.split('_')[2])
        #print(oldlabel)
        if newlabel==oldlabel== 0:
            tempfilename = ("1_" + tempfilename)
        elif newlabel==oldlabel== 1:
            tempfilename = ("2_" + tempfilename)
        elif newlabel==oldlabel== 2:
            tempfilename = ("3_" + tempfilename)
        elif newlabel==oldlabel== 3:
            tempfilename = ("0_" + tempfilename)
        '''
        i = i + 1
        j = j + 1

        # savepath为处理后文件保存的全路径
        savepath = os.path.join(img_savepath, tempfilename)
        cv2.imwrite(savepath, img)
