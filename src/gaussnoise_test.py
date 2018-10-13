# coding:utf-8

import numpy as np
from libtiff import TIFF
path = 'D:/图集/gaussnoise_test/test1.tif'
imgdir = TIFF.open(path, mode="r")
imgarr = imgdir.read_image()
noise_add = np.random.normal(size=(145, 145))
for i in range(145):
    for j in range(145):
        imgarr[i][j] += noise_add[i][j] * 200
        if(imgarr[i][j] > 65535):
            imgarr[i][j] = 65535
        elif(imgarr[i][j] < 0):
            imgarr[i][j] = 0

imgname = 'D:/图集/gaussnoise_test/' + '_' + 'sample1.tif'
img = TIFF.open(imgname, 'w')
img.write_image(imgarr)
print('success!')
