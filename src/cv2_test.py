import numpy as np
from libtiff import TIFF

if __name__ == '__main__':
    path = 'D:/aviris_hyperspectral_data/19920612_AVIRIS_IndianPine_EW-line_R_Gaussnoise.tif'
    imgarr = np.zeros((1848, 614, 220), dtype=np.uint16)
    imgarr = np.transpose(TIFF.open(path, mode="r").read_image())
    newimgarr = np.zeros((200, 200, 220), dtype=np.uint16)

    for i in range(220):
        print('success! %d' % i)
        for j in range(200):
            for k in range(200):
                newimgarr[k, j, i] = (imgarr[k][j][i] / 9 + imgarr[k + 1][j][i] / 9 + imgarr[k + 2][j][i] / 9
                                      + imgarr[k][j + 1][i] / 9 + imgarr[k +
                                                                         1][j + 1][i] / 9 + imgarr[k + 2][j + 1][i] / 9
                                      + imgarr[k][j + 2][i] / 9 + imgarr[k + 1][j + 2][i] / 9 + imgarr[k + 2][j + 2][i] / 9)

    newimgname = 'D:/aviris_hyperspectral_data/Gaussnoise_GaussianAve_200.tif'
    img = TIFF.open(newimgname, 'w')
    img.write_image(np.transpose(newimgarr), write_rgb=True)
    print('success!')

    for i in range(3):
        for j in range(2):
            for k in range(5):
                print(newimgarr[k, j, i], imgarr[k, j, i])

    '''
    print(type(imgarr[1, 1, 1]))
    for i in range(2):
        for j in range(12):
            for k in range(6):
                print(imgarr[k][j][i] + imgarr[k + 1][j][i] + imgarr[k + 2][j][i] + imgarr[k][j + 1][i] + imgarr[k + 1]
                      [j + 1][i] + imgarr[k + 2][j + 1][i] + imgarr[k][j + 2][i] + imgarr[k + 1][j + 2][i] + imgarr[k + 2][j + 2][i])
                      '''
