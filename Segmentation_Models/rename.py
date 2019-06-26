# -*- coding: utf-8 -*-
"""
Created on 2019-3-12
@author: LeonShangguan
"""
import os
import cv2

cnt = 0
for data_file in sorted(os.listdir(os.getcwd() + '/data/train/label')):
    cnt = cnt + 1

    img = cv2.imread(os.getcwd() + '/data/train/label/'+ data_file)
    cv2.imwrite(os.getcwd() + '/data/train/aug/label/' + str(cnt) + 'o.png', img)

    hImg = cv2.flip(img,1,dst=None) #水平镜像
    cv2.imwrite(os.getcwd() + '/data/train/aug/label/' + str(cnt)  + 'h.png', hImg)
    vImg = cv2.flip(img,0,dst=None) #垂直镜像
    cv2.imwrite(os.getcwd() + '/data/train/aug/label/' + str(cnt) + 'v.png', vImg)
    cImg = cv2.flip(img,-1,dst=None) #对角镜像
    cv2.imwrite(os.getcwd() + '/data/train/aug/label/' + str(cnt) + 'c.png', cImg)
    
    print(data_file)

print('************************************************************************')

# cnt = 0

# for data_file in sorted(os.listdir(path + 'label')):
#     cnt = cnt + 1

#     img = cv2.imread('label/' + data_file)
#     cv2.imwrite('aug/label/' + str(cnt) + 'o.png', img)

#     hImg = cv2.flip(img,1,dst=None) #水平镜像
#     cv2.imwrite('aug/label/' + str(cnt) + 'h.png', hImg)
#     vImg = cv2.flip(img,0,dst=None) #垂直镜像
#     cv2.imwrite('aug/label/' + str(cnt) + 'v.png', vImg)
#     cImg = cv2.flip(img,-1,dst=None) #对角镜像
#     cv2.imwrite('aug/label/' + str(cnt) + 'c.png', cImg)

#     print(data_file)
