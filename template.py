import cv2 as cv
import numpy as numpy
from  matplotlib import pyplot as plt 

img = cv.imread('opencv1.png')

img_gray = cv.cvtColor(img,cv.COLOR_BGRA2GRAY)

ret, thresh1 = cv.threshold(img,4,255,cv.THRESH_BINARY)
ret, thresh2 = cv.threshold(img,4,255,cv.THRESH_BINARY_INV)
ret, thresh3 = cv.threshold(img,4,255,cv.THRESH_TRUNC)
ret, thresh4 = cv.threshold(img,200,255,cv.THRESH_TOZERO)
ret, thresh5 = cv.threshold(img,200,255,cv.THRESH_TOZERO_INV)

titulos = ["original","Binary", "Binary_inv", "Trunc", "ToZero","ToZero_inv"]
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i],"gray")
    plt.title(titulos[i])
    plt.xticks([]),plt.yticks([])

plt.show()