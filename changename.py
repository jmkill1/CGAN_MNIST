# -*- coding: utf-8 -*-
# @Time : 2021/7/3 1:02 下午
# @Author : Zijie Huang 
# @FileName: changename.py
# @Email : 23020201153760@stu.xmu.edu.cn
# @Software: PyCharm
import os
import tensorflow as tf
import numpy as np

def rename():
    path='./out/overwriting/49/64/4/'
    filelist=os.listdir(path)
    for f in filelist:
        olddir=os.path.join(path,f)
        if(os.path.isdir(olddir)):
            continue
        fn=os.path.splitext(f)[0]
        ft=os.path.splitext(f)[1]
        newdir=os.path.join(path,fn+"d"+ft)
        os.rename(olddir,newdir)

rename()