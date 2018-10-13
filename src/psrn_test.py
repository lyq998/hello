# coding:utf-8
from skimage import measure
from libtiff import TIFF

path_images = 'F:/SCUinnovation/test_image/images/'
path_labels = 'F:/SCUinnovation/test_image/labels/'

index = 0
psnr_all = 0
mse_all = 0

for i in range(4838):
    image_path = path_images + str(i) + '.tif'
    label_path = path_labels + str(i) + '.tif'
    imgdir0 = TIFF.open(image_path, mode="r")
    imgarr0 = imgdir0.read_image()
    imgdir1 = TIFF.open(label_path, mode="r")
    imgarr1 = imgdir1.read_image()

    psnr_all = psnr_all + measure.compare_psnr(imgarr1, imgarr0, 65535)
#    mse_all = mse_all + measure.compare_mse(imgarr1, imgarr0)

print(psnr_all / 4838)
