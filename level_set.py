import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.ndimage


'''
This codes try to implement level set algorithm.

'''

dt = 1
it = 1000
sigma = 20

img = cv2.imread("Image\KJB\IMG_2224.JPG")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_smooth = scipy.ndimage.filters.gaussian_filter(gray, sigma)

dphi_x, dphi_y = np.gradient(img_smooth)
dphi_pow = np.square(dphi_x) + np.square(dphi_y)
Force = 1. / (1. + dphi_pow)

for i in range(it):
    print(i)
    dphi_x, dphi_y = np.gradient(img_smooth)
    dphi = np.square(dphi_x) + np.square(dphi_y)
    dphi_norm = np.sqrt(dphi)
    Force = 1. / (1. + dphi)

    img_smooth = img_smooth + dt * Force * dphi_norm

img_show = img_smooth.copy()
img_show = np.uint8(img_show)
img_show[img_show > 150] = 255
img_show[img_show <= 150] = 0

(cnts,_) = cv2.findContours(img_show, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
area = np.array([cv2.contourArea(cnts[i]) for i in range(len(cnts))])
maxa_ind = np.argmax(area)
img_contour = img.copy()
cv2.drawContours(img_contour, cnts[maxa_ind], -1, (0,255,0), 20)
b,g,r = cv2.split(img_contour)  
img_contour = cv2.merge([r,g,b])  
plt.imshow(img_contour)
plt.show()
