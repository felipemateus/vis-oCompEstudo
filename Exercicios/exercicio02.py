import cv2 as cv
import time
import numpy as np

img = cv.imread("lena_cor.jpg")
count = 0;
while True:
    #atualizo a imagem
    cv.imshow("image",img)
    time.sleep(10000)
    count += 1
    if count == 10:
        break 





cv.waitKey(0)
cv.destroyAllWindows()