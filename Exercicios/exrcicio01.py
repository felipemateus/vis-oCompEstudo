import cv2 as cv
import numpy as np

size = 10
img = cv.imread("lena_cor.jpg")
img[0:size , 0:size] = 255





cv.imshow("image",img)
cv.waitKey(0)
cv.destroyAllWindows()