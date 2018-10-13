# coding:utf-8
import numpy as np
from libtiff import TIFF
path0 = 'F:/SCUinnovation/aviris_hyperspectral_data/19920612_AVIRIS_IndianPine_EW-line_R.tif'
path1 = 'F:/SCUinnovation/aviris_hyperspectral_data/19920612_AVIRIS_IndianPine_NS-line.tif'
path2 = 'F:/SCUinnovation/aviris_hyperspectral_data/19920612_AVIRIS_IndianPine_Site3.tif'
noise_path0 = 'F:/SCUinnovation/aviris_hyperspectral_data/19920612_AVIRIS_IndianPine_EW-line_R_Gaussnoise.tif'
noise_path1 = 'F:/SCUinnovation/aviris_hyperspectral_data/19920612_AVIRIS_IndianPine_NS-line_Gaussnoise.tif'
noise_path2 = 'F:/SCUinnovation/aviris_hyperspectral_data/19920612_AVIRIS_IndianPine_Site3_Gaussnoise.tif'

savepath0 = 'F:/SCUinnovation/train_image/images/'
savepath1 = 'F:/SCUinnovation/train_image/labels/'
savepath2 = 'F:/SCUinnovation/test_image/images/'
savepath3 = 'F:/SCUinnovation/test_image/labels/'

#imgdir0 = TIFF.open(path0, mode="r")
#imgdir1 = TIFF.open(path1, mode="r")
#imgdir2 = TIFF.open(path2, mode="r")
#imgdir3 = TIFF.open(noise_path0, mode="r")
imgdir4 = TIFF.open(noise_path1, mode="r")
#imgdir5 = TIFF.open(noise_path2, mode="r")

#imgarr0 = imgdir0.read_image()
#imgarr1 = imgdir1.read_image()
#imgarr2 = imgdir2.read_image()
#imgarr3 = imgdir3.read_image()
imgarr4 = imgdir4.read_image()
#imgarr5 = imgdir5.read_image()

index = 5900
# 当为图一加图二时，初值为5900
save_arr = np.zeros((220, 30, 30), dtype=np.uint16)


def copy_img(row, col, imgarr):
    for k in range(220):
        for i in range(30):
            for j in range(30):
                save_arr[k][i][j] = imgarr[k][i + row][j + col]


'''
i,j的循环次数要自己计算出边界
分别是：i       j
             59    182
            265    59
             12       12 
由于第三个图太小故不用，将第一张图分成两份，前59*82作为test_image
后面59*100的加上图二一起作为train_image
'''
for i in range(265):
    for j in range(59):
        copy_img(i * 10, j * 10, imgarr4)
        save_imgname = savepath0 + str(index) + '.tif'
        img = TIFF.open(save_imgname, 'w')
        img.write_image(save_arr, write_rgb=True)
        index = index + 1
        if index % 100 == 0:
            print('%d     success!' % index)

'''
print(imgarr2.dtype)
for i in range(3):
    print(imgarr2.shape[i])
'''
# 查看原矩阵数据类型
