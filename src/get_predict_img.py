# coding:utf-8
import numpy as np
import time
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Activation, Dense, Reshape
from keras.layers import Conv2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import RMSprop, Adam
from libtiff import TIFF

savepath0 = 'F:/SCUinnovation/train_image/images/'
savepath1 = 'F:/SCUinnovation/train_image/labels/'
savepath2 = 'F:/SCUinnovation/test_image/images/'
savepath3 = 'F:/SCUinnovation/test_image/labels/'
savepath4 = 'F:/SCUinnovation/train_image/gan_train_images/'
savepath5 = 'F:/SCUinnovation/test_image/predict_img/'
savepath6 = 'D:/aviris_hyperspectral_data/100img/'
savepath7 = 'D:/aviris_hyperspectral_data/100img_pre/'

#modelpath = 'F:/SCUinnovation/my_model.h5'


class ElapsedTimer(object):
    def __init__(self):
        self.start_time = time.time()

    def elapsed(self, sec):
        if sec < 60:
            return str(sec) + " sec"
        elif sec < (60 * 60):
            return str(sec / 60) + " min"
        else:
            return str(sec / (60 * 60)) + " hr"

    def elapsed_time(self):
        print("Elapsed: %s " % self.elapsed(time.time() - self.start_time))


class CNN_5_PILES(object):
    def __init__(self, img_rows=30, img_cols=30, channel=220):

        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channel = channel
        self.CNN_5 = None   # generator

    def cnn_5(self):
        if self.CNN_5:
            return self.CNN_5
        self.CNN_5 = Sequential()
        dropout = 0.4
        depth = 128
        # In: 30*30*220
        # Out:30*30*32
        input_shape = (self.img_rows, self.img_cols, self.channel)
        self.CNN_5.add(Conv2D(depth * 2, 3, strides=1, input_shape=input_shape,
                              padding='same'))
        self.CNN_5.add(BatchNormalization(momentum=0.9))
        self.CNN_5.add(Activation('relu'))
        self.CNN_5.add(Dropout(dropout))

        self.CNN_5.add(Conv2D(depth * 2, 3, strides=1, padding='same'))
        self.CNN_5.add(BatchNormalization(momentum=0.9))
        self.CNN_5.add(Activation('relu'))
        self.CNN_5.add(Dropout(dropout))

        self.CNN_5.add(Conv2D(depth * 2, 3, strides=1, padding='same'))
        self.CNN_5.add(BatchNormalization(momentum=0.9))
        self.CNN_5.add(Activation('relu'))
        self.CNN_5.add(Dropout(dropout))

        self.CNN_5.add(Conv2D(220, 3, strides=1, padding='same'))
#        self.CNN_5.add(Activation('sigmoid'))
        self.CNN_5.summary()
#        optimizer = RMSprop(lr=0.0004, decay=3e-8)
        optimizer = Adam(lr=0.1)
        self.CNN_5.compile(loss='mean_squared_error', optimizer=optimizer,
                           metrics=['accuracy'])
        return self.CNN_5


class CNN_MODEL(object):
    def __init__(self):
        self.CNN_5_PILES = CNN_5_PILES()
        self.model = self.CNN_5_PILES.cnn_5()

    def train(self, train_steps=2000, batch_size=256, save_interval=0):
        imgdir_img = np.zeros((batch_size, 30, 30, 220), dtype=np.uint16)
        imgdir_lab = np.zeros((batch_size, 30, 30, 220), dtype=np.uint16)
        for i in range(train_steps):
            for num in range(batch_size):
                random_int = np.random.randint(0, 21534)
                imgdir_img[num] = np.transpose(TIFF.open(
                    savepath0 + str(random_int) + '.tif', mode="r").read_image())
                imgdir_lab[num] = np.transpose(TIFF.open(
                    savepath1 + str(random_int) + '.tif', mode="r").read_image())
            loss = self.model.train_on_batch(imgdir_img, imgdir_lab)
            log_mesg = "%d: [loss: %f, acc: %f]" % (i, loss[0], loss[1])
            if (i + 1) % 10 == 0:
                print(log_mesg)

            if save_interval > 0:
                if (i + 1) % save_interval == 0:
                    self.model.save(
                        'F:/SCUinnovation/model/CNN_4_test1/my_model_test1.h5')

    def load(self):
        self.model = load_model(
            'F:/SCUinnovation/model/CNN_4_test1/my_model_test1.h5')


if __name__ == '__main__':
    cnn = CNN_MODEL()
    cnn.load()
    index1 = 0
    index2 = 0
    save_arr_batch = np.zeros((128, 30, 30, 220), dtype=np.uint16)
    in_arr_batch = np.zeros((128, 30, 30, 220), dtype=np.uint16)
    save_arr = np.zeros((220, 30, 30), dtype=np.uint16)
    in_arr = np.zeros((220, 30, 30), dtype=np.uint16)
    for j in range(128):
        in_imgname = savepath6 + str(index1) + '.tif'
        in_arr = TIFF.open(in_imgname, mode="r").read_image()
        in_arr_batch[j] = np.transpose(in_arr)
        index1 = index1 + 1

    save_arr_batch = cnn.model.predict(in_arr_batch).astype(np.uint16)

    for k in range(128):
        save_arr = np.transpose(save_arr_batch[k])
        save_imgname = savepath7 + str(index2) + '.tif'
        img = TIFF.open(save_imgname, 'w')
        img.write_image(save_arr, write_rgb=True)
        index2 = index2 + 1
        if index2 % 500 == 0:
            print('%d     success!' % index2)
