import cv2
import numpy as np 

img = cv2.imread('bookpage.jpg')

grayscaled = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#retval, threshold = cv2.threshold(grayscaled, 11, 255 , cv2.THRESH_BINARY)
threshold = cv2.adaptiveThreshold(grayscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)

median = cv2.medianBlur(threshold,3)
gaus = cv2.GaussianBlur(threshold,(5,5),0)



cv2.imshow('original',img)
cv2.imshow('gray',grayscaled)
cv2.imshow('threshold',threshold)
cv2.imshow('gaus',gaus)

cv2.waitKey(0)
cv2.destroyAllWindows()