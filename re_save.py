# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 17:28:32 2021

@author: laconicli
"""

import os 
import shutil
import cv2
SIZE_of_Img = (512, 512)
class name_count():
    def __init__(self):
        self.cnt=0
    def get_cnt(self,):
        self.cnt += 1
        return self.cnt

def file_name(file_dir,file_type=''):#默认为文件夹下的所有文件
    lst = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if(file_type == ''):
                lst.append(os.path.join(root,file))
            else:
                if os.path.splitext(file)[1] == str(file_type):#获取指定类型的文件名
                    lst.append(os.path.join(root,file))
    return lst

def split_img(file, check):
    target=[]
    img=[]
    for path, c in zip(file, check):
        if c:
            target.append(path.replace('\\','/'))
        else:
            img.append(path.replace('\\','/'))
    return set(img), set(target)

def make_dict(img, tgt):
    path_dict=dict()
    for t in tgt:
        root, name = t.split('/GroundTruth/')
        n, _ = name.split('.')
        path = [os.path.join(root, n+'.jpg').replace('\\','/'), os.path.join(root, n+'.bmp').replace('\\','/'), os.path.join(root, n+'.png').replace('\\','/')]
        for p in path:
            if p in img:
                path_dict[p] = t
                break
    return path_dict

def rebuild(path_dict, folder='data'):
    cnt = name_count()
    keys=path_dict.keys()
    for k in keys:
        img, tgt = k, path_dict[k]
        
        # get the count number
        c = cnt.get_cnt()
        
        jpg_img, png_tgt = cv2.imread(img), cv2.imread(tgt)
        jpg_img, png_tgt = cv2.resize(jpg_img, SIZE_of_Img), cv2.resize(png_tgt, SIZE_of_Img)
        png_tgt = 1*(png_tgt[:,:,0]>0)
        # shutil.copy(img, os.path.join(folder, 'train_val', '%06d' % c + '.jpg'))
        # shutil.copy(tgt, os.path.join(folder, 'val', '%06d' % c + '.png'))
        
        cv2.imwrite(os.path.join(folder, 'train_val', '%06d' % c + '.jpg'), jpg_img)
        cv2.imwrite(os.path.join(folder, 'val', '%06d' % c + '.png'), png_tgt)
        
if __name__=="__main__":
    file = file_name(r'C:/Users/laconicli/Downloads/ObjectDiscovery-data')
    check = ["GroundTruth" in f for f in file]
    img, tgt = split_img(file, check)
    del file, check
    path_dict = make_dict(img, tgt)
    rebuild(path_dict)

