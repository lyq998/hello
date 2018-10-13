import numpy as np
from libtiff import TIFF

path = 'D:/aviris_hyperspectral_data/100img_pre/'

save_arr = np.zeros((220, 300, 300), dtype=np.uint16)


def copy_img(row, col, imgarr):
    for k in range(220):
        for i in range(30):
            for j in range(30):
                save_arr[k][i + row][j + col] = imgarr[k][i][j]


if __name__ == '__main__':

    for m in range(10):
        for n in range(10):
            imgarr = TIFF.open(path + str(10 * m + n) +
                               '.tif', mode="r").read_image()
            copy_img(m * 30, n * 30, imgarr)

    save_imgname = 'D:/aviris_hyperspectral_data/DeepLearn_300.tif'
    img = TIFF.open(save_imgname, 'w')
    img.write_image(save_arr, write_rgb=True)

    print('success!')
