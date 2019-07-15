import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import cv2
 

img = cv2.imread("watch.jpg",cv2.IMREAD_GRAYSCALE)


# cv2.imshow("image",img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

plt.imshow(img, cmap='gray', interpolation = 'bicubic')
plt.show()