# coding:GBK
import numpy as np
from libtiff import TIFF
from skimage import measure

imgpath_original = 'D:/aviris_hyperspectral_data/19920612_AVIRIS_IndianPine_EW-line_R.tif'
imgpath_deeplearning = 'D:/aviris_hyperspectral_data/DeepLearn_300.tif'
imgpath_midfil = 'D:/aviris_hyperspectral_data/Gaussnoise_Midfil_200.tif'
imgpath_avefil = 'D:/aviris_hyperspectral_data/Gaussnoise_GaussianAve_200.tif'
imgpath_gaussian_seta8 = 'D:/aviris_hyperspectral_data/Gaussnoise_Gaussian_Seta0.8_200.tif'


# 拷贝original_imgarr中的n*n到des_imgarr中，通道数为220
def copy_img_n(n, original_imgarr, des_imgarr):
    for k in range(220):
        for j in range(n):
            for i in range(n):
                des_imgarr[k, j, i] = original_imgarr[k, j, i]


if __name__ == '__main__':
    img_original_arr_200 = np.zeros((220, 200, 200), dtype=np.uint16)
    img_original_arr_300 = np.zeros((220, 300, 300), dtype=np.uint16)

    img_original_arr = TIFF.open(imgpath_original, mode="r").read_image()
    copy_img_n(200, img_original_arr, img_original_arr_200)
    copy_img_n(300, img_original_arr, img_original_arr_300)

    img_deeplearning_arr = TIFF.open(
        imgpath_deeplearning, mode="r").read_image()
    img_midfil_arr = TIFF.open(imgpath_midfil, mode="r").read_image()
    img_avefil_arr = TIFF.open(imgpath_avefil, mode="r").read_image()
    img_gaussian_seta8 = TIFF.open(
        imgpath_gaussian_seta8, mode="r").read_image()

    mse_deeplearning = measure.compare_mse(
        img_original_arr_300, img_deeplearning_arr)
    mse_midfil = measure.compare_mse(img_original_arr_200, img_midfil_arr)
    mse_avefil = measure.compare_mse(img_original_arr_200, img_avefil_arr)
    mse_gussian_seta8 = measure.compare_mse(
        img_original_arr_200, img_gaussian_seta8)

    print('mse_deeplearning:', mse_deeplearning)
    print('mse_midfil:', mse_midfil)
    print('mse_avefil:', mse_avefil)
    print('mse_gussian_seta8:', mse_gussian_seta8)
