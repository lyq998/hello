# coding:utf-8
from libtiff import TIFF

i = 0
imgname0 = 'E:/SCUinnovation/aviris_hyperspectral_data/19920612_AVIRIS_IndianPine_Site3.tif'
imgname1 = 'E:/SCUinnovation/aviris_hyperspectral_data/19920612_AVIRIS_IndianPine_Site3_Gaussnoise1.tif'
img0 = TIFF.open(imgname0, 'r')
img1 = TIFF.open(imgname1, 'r')
imgarr0 = img0.read_image()
imgarr1 = img1.read_image()

for i in range(3):
    print('----------------------------')
    for j in range(10):
        print(str(imgarr0[i][j][0]) + ' ' + str(imgarr1[i][j][0]))
