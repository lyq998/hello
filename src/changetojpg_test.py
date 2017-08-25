from PIL import Image
import cv2

#im = Image.open('D:/aviris_hyperspectral_data/19920612_AVIRIS_IndianPine_EW-line_R_Gaussnoise.tif')

# im.save('D:/aviris_hyperspectral_data/19920612_AVIRIS_IndianPine_EW-line_R_Gaussnoise.jpg')

read_img_name = 'D:/aviris_hyperspectral_data/19920612_AVIRIS_IndianPine_EW-line_R_Gaussnoise.tif'
img = cv2.imread(read_img_name)
file_name = 'D:/aviris_hyperspectral_data/19920612_AVIRIS_IndianPine_EW-line_R_Gaussnoise.jpg'
#cv2.imwrite(file_name, img)
print(type(img).__name__)
print(img.dtype)
