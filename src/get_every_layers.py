# coding:GBK
import numpy as np
from libtiff import TIFF

imgpath_original = 'D:/aviris_hyperspectral_data/19920612_AVIRIS_IndianPine_EW-line_R.tif'
imgpath_noise = 'D:/aviris_hyperspectral_data/19920612_AVIRIS_IndianPine_EW-line_R_Gaussnoise.tif'
imgpath_deeplearning = 'D:/aviris_hyperspectral_data/DeepLearn_300.tif'
imgpath_midfil = 'D:/aviris_hyperspectral_data/Gaussnoise_Midfil_300.tif'
imgpath_avefil = 'D:/aviris_hyperspectral_data/Gaussnoise_GaussianAve_300.tif'
imgpath_gaussian_seta8 = 'D:/aviris_hyperspectral_data/Gaussnoise_Gaussian_Seta0.8_300.tif'
imgpath_bm4d = 'D:/aviris_hyperspectral_data/mat_positive/bm4d_10to50.tif'

save_path_original = 'D:/aviris_hyperspectral_data/all_layers_original/'
save_path_noise = 'D:/aviris_hyperspectral_data/all_layers_noise/'
save_path_deeplearning = 'D:/aviris_hyperspectral_data/all_layers_deeplearning/'
save_path_midfil = 'D:/aviris_hyperspectral_data/all_layers_midfil/'
save_path_avefil = 'D:/aviris_hyperspectral_data/all_layers_avefil/'
save_path_gaussian_seta8 = 'D:/aviris_hyperspectral_data/all_layers_gaussian_seta8/'
save_path_bm4d = 'D:/aviris_hyperspectral_data/10to50layers_bm4d/'

if __name__ == '__main__':
    img_original_arr = TIFF.open(imgpath_original, mode="r").read_image()
    img_noise_arr = TIFF.open(imgpath_noise, mode="r").read_image()
    img_deeplearning_arr = TIFF.open(
        imgpath_deeplearning, mode="r").read_image()
    img_midfil_arr = TIFF.open(imgpath_midfil, mode="r").read_image()
    img_avefil_arr = TIFF.open(imgpath_avefil, mode="r").read_image()
    img_gaussian_seta8_arr = TIFF.open(
        imgpath_gaussian_seta8, mode="r").read_image()
    img_bm4d_arr = TIFF.open(imgpath_bm4d, mode="r").read_image()
    '''
    #下面是原始图像的每一层
    for i in range(220):
        save_imgname = save_path_original + str(i) + '.tif'
        img = TIFF.open(save_imgname, 'w')
        img.write_image(img_original_arr[i], write_rgb=True)

    # 下面是噪声图像的每一层
    for i in range(220):
        save_imgname = save_path_noise + str(i) + '.tif'
        img = TIFF.open(save_imgname, 'w')
        img.write_image(img_noise_arr[i], write_rgb=True)
    
    # 下面是得到深度神经网络后的每一层
    for i in range(220):
        save_imgname = save_path_deeplearning + str(i) + '.tif'
        img = TIFF.open(save_imgname, 'w')
        img.write_image(img_deeplearning_arr[i], write_rgb=True)
    
    # 下面是得到midfil后的每一层
    for i in range(220):
        save_imgname = save_path_midfil + str(i) + '.tif'
        img = TIFF.open(save_imgname, 'w')
        img.write_image(img_midfil_arr[i], write_rgb=True)
    
    # 下面是得到avefil后的每一层
    for i in range(220):
        save_imgname = save_path_avefil + str(i) + '.tif'
        img = TIFF.open(save_imgname, 'w')
        img.write_image(img_avefil_arr[i], write_rgb=True)
    
    # 下面是得到avefil后的每一层
    for i in range(220):
        save_imgname = save_path_gaussian_seta8 + str(i) + '.tif'
        img = TIFF.open(save_imgname, 'w')
        img.write_image(img_gaussian_seta8_arr[i], write_rgb=True)
    '''
    # 下面是得到bm4d后的10到50层图像
    for i in range(41):
        save_imgname = save_path_bm4d + str(i + 10) + '.tif'
        img = TIFF.open(save_imgname, 'w')
        img.write_image(img_bm4d_arr[i], write_rgb=True)
