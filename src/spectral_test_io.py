# -*- coding:gbk -*-
from libtiff import TIFF
from scipy import misc
#import matplotlib.pyplot as plt

path = 'E:/SCUinnovation/aviris_hyperspectral_data/19920612_AVIRIS_IndianPine_Site3_Gaussnoise.tif'


def tiff_to_image_array(tiff_image_name, out_folder, out_type):

    tif = TIFF.open(tiff_image_name, mode="r")
    idx = 0
    for im in list(tif.iter_images()):
        im_name = out_folder + str(idx) + out_type
        tiff = TIFF.open(im_name, mode='w')
        tiff.write_image(im)
        print(im_name, 'successfully saved!!!')
        idx = idx + 1
    return


if __name__ == '__main__':
    tiff_to_image_array(
        path, 'E:/SCUinnovation/aviris_hyperspectral_data', '.jpg')
