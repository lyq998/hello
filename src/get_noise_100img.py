import numpy as np
from libtiff import TIFF

save_arr = np.zeros((30, 30, 220), dtype=np.uint16)


def copy_img(row, col, imgarr):
    for k in range(220):
        for i in range(30):
            for j in range(30):
                save_arr[i][j][k] = imgarr[i + row][j + col][k]


if __name__ == '__main__':
    path = 'D:/aviris_hyperspectral_data/19920612_AVIRIS_IndianPine_EW-line_R_Gaussnoise.tif'
    savepath = 'D:/aviris_hyperspectral_data/100img/'

    imgarr = np.zeros((1848, 614, 220), dtype=np.uint16)
    imgarr = np.transpose(TIFF.open(path, mode="r").read_image())
    index = 0
    for m in range(13):
        for n in range(10):
            copy_img(30 * n, 30 * m, imgarr)
            save_imgname = savepath + str(index) + '.tif'
            img = TIFF.open(save_imgname, 'w')
            img.write_image(np.transpose(save_arr), write_rgb=True)
            index = index + 1
