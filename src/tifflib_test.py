# coding:utf-8
'''
Created on 2018.4.13

@author: lyq
'''
from libtiff import TIFF
path = 'D:/图集/aviris_hyperspectral_data/19920612_AVIRIS_IndianPine_Site3.tif'
imgdir = TIFF.open(path, mode="r")
imgarr = imgdir.read_image()
for i in range(3):
    print(imgarr.shape[i])
print(imgarr[45])

'''
for i in range(imgarr.shape[0]):
    imgname = path + "_" + str(i) + ".tif"
    img = TIFF.open(imgname, 'w')
    img.write_image(imgarr[i])
'''
