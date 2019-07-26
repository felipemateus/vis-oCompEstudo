import numpy as np
import cv2


img = cv2.imread('watch.jpg',cv2.IMREAD_ANYCOLOR)

px = img[5,100]

img[100:150, 100:150] = [255,255,255]

watch_face = img[37:111, 107:194]


img[0:74, 0:87] = watch_face





cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()



print(px)

