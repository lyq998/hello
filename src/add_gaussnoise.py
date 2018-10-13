# coding:utf-8
import numpy as np
from libtiff import TIFF
path0 = 'E:/SCUinnovation/aviris_hyperspectral_data/19920612_AVIRIS_IndianPine_EW-line_R.tif'
path1 = 'E:/SCUinnovation/aviris_hyperspectral_data/19920612_AVIRIS_IndianPine_NS-line.tif'
path2 = 'E:/SCUinnovation/aviris_hyperspectral_data/19920612_AVIRIS_IndianPine_Site3.tif'
imgdir0 = TIFF.open(path0, mode="r")
#imgdir1 = TIFF.open(path1, mode="r")
#imgdir2 = TIFF.open(path2, mode="r")
imgarr0 = imgdir0.read_image()
#imgarr1 = imgdir1.read_image()
#imgarr2 = imgdir2.read_image()
# imgarr的shape[0][1][2]分别对应通道，宽，长，反过来的
#noise_add0 = np.random.normal(size=(220, 614, 1848))
#noise_add1 = np.random.normal(size=(220, 2678, 614))
#noise_add2 = np.random.normal(size=(220, 145, 145))
# np.random.normal()只能创建二维正态分布

for i in range(220):
    noise_add0 = np.random.normal(size=(614, 1848))
    #noise_add2 = np.random.normal(size=(145, 145))
    for j in range(614):
        for k in range(1848):
            imgarr0[i][j][k] += noise_add0[j][k] * 200
            if(imgarr0[i][j][k] > 65535):
                imgarr0[i][j][k] = 65535
            elif(imgarr0[i][j][k] < 0):
                imgarr0[i][j][k] = 0

imgname = 'E:/SCUinnovation/aviris_hyperspectral_data/19920612_AVIRIS_IndianPine_EW-line_R_Gaussnoise.tif'
img = TIFF.open(imgname, 'w')
img.write_image(imgarr0, write_rgb=True)
print('success!')
