from libtiff import TIFF
import cv2
import scipy.io as scio
import numpy as np

'''
file_path = 'D:/aviris_hyperspectral_data/19920612_AVIRIS_IndianPine_EW-line_R_Gaussnoise.tif'
save_path = 'D:/aviris_hyperspectral_data/19920612_AVIRIS_IndianPine_EW-line_R_Gaussnoise.jpg'

img_arr = TIFF.open(file_path, mode="r").read_image()
print(img_arr.shape)
#cv2.imwrite(save_path, img_arr)
max = 0
for i in range(220):
    for j in range(614):
        for k in range(1848):
            if(max < img_arr[i, j, k]):
                max = img_arr[i, j, k]

print(max)

# max=65531
'''

dataFile = 'y_est.mat'
data = scio.loadmat(dataFile)

# print(data['y_est'])
print(data['y_est'].shape)

img_arr = np.transpose(data['y_est'])
print(img_arr.shape)

save_path = 'D:/aviris_hyperspectral_data/bm4d_10to50.tif'
img = TIFF.open(save_path, 'w')
img.write_image(img_arr, write_rgb=True)
