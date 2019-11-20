import matplotlib
matplotlib.use('TkAgg')

import cv2
import numpy as np
import time
from matplotlib import pyplot as plt

def main():

    #img= cv2.imread('primeiroFrame.jpg',0)
    img= cv2.imread('primeiroFrame.jpg',cv2.IMREAD_COLOR)
    print(img.shape)
    #nomaliza a imagem
    img = histogramaNormalizadoColor(img)

    #aplica filtro gausiano
    img = cv2.GaussianBlur(img,(5,5),0)
    
    #calcula os clauster usando kmeans:
    img = calculakmeans(img)
    
    
    #aplica trashholding separando as cores vermelhas
    mask = trashholdingVermelho(img)

    #aplica uma operação de closing e open na mascara para remover os ruidos
    retKernel = cv2.getStructuringElement(cv2.MORPH_RECT,(50,1))
    openImage = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,retKernel)
    retKernel = cv2.getStructuringElement(cv2.MORPH_RECT,(20,10))
    closeImage = cv2.morphologyEx(openImage,cv2.MORPH_OPEN,retKernel)
    retKernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    openImage = cv2.morphologyEx(closeImage,cv2.MORPH_CLOSE,retKernel)

    #encontra os blobs:
    #img_blob = deteccaoDeBlobs(openImage)

    #utilizando momento:
    img_blob = detectMomento(openImage)

    #cv2.imshow("Final",openImage)
    #cv2.imshow("CloseiImage",closeImage)

    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    

    #cap.release()




def detectMomento(img):
    #img= cv2.imread('untitled.png',0)

    ret,thresh = cv2.threshold(img,125,255,0)
    countourn, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


    countourn = list(reversed(countourn)) 
    cnt = countourn[0]
    M = cv2.moments(cnt)
    cx = int(M["m10"]/M["m00"])
    cy = int(M["m01"]/M["m00"])
    print(cx)
    print(cy)
    x,y,w,h = cv2.boundingRect(countourn[0])
    cv2.rectangle(img,(x-10,y-10), (x+w+10,y+h+10),(255,255,255),1)
   
    cv2.imshow("img_contours", img)
    cv2.waitKey(0) 

"""     for c in countourn:
        M = cv2.moments(c)
        cx = int(M["m10"]/M["m00"])
        cy = int(M["m01"]/M["m00"])
        cv2.circle(img, (cX, cY), 5, (255, 255, 255), -1)
        cv2.imshow("img_contourn", img)

    print(countourn)
    cv2.drawContours(img, countourn, -1, (0,255,0), 3)
    cv2.imshow("img_contours", img)
    cv2.waitKey(0) """
   









def deteccaoDeBlobs(img):
    #img= cv2.imread('untitled.png',cv2.IMREAD_GRAYSCALE)
    params = cv2.SimpleBlobDetector_Params()
    #img = cv2.bitwise_not(img)
    

    params.minDistBetweenBlobs = 10 
    params.filterByColor = True
    params.blobColor = 255
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False

    params.filterByArea = True # 
    params.minArea = 1 # 
    params.maxArea = 100000 # 
    
    
    
    # Create a detector with the parameters
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3 :
        detector = cv2.SimpleBlobDetector(params)
    else : 
        detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(img)
    print(type(keypoints))
    keypoints = list(reversed(keypoints)) 
     #np.invert(keypoints)
    for i in keypoints:
        im_with_keypoints = cv2.drawKeypoints(img, [i], np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        print("################")
        print(i.class_id)
        print(i.pt)
        objectCentroid = i.pt
        print("################")
        break
      
    black =  np.zeros((540,960,3))


    #print(im_with_keypoints.size)

    
    #im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow("Keypoints", im_with_keypoints)
    cv2.waitKey(0)





def trashholdingVermelho(colorImage):
    #converte para HSV
    hsv =  cv2.cvtColor(colorImage, cv2.COLOR_BGR2HSV)
    
    lower_red = np.array([0,70,50])
    upper_red = np.array([10,255,255])
    mask1 = cv2.inRange(hsv,lower_red,upper_red)

    lower_red = np.array([170,70,50])
    upper_red = np.array([180,255,255])
    mask2 = cv2.inRange(hsv,lower_red,upper_red)

    mask = mask1 | mask2

    res = cv2.bitwise_and(colorImage,colorImage, mask = mask)
    #cv2.imshow('frame',colorImage)
    #cv2.imshow('mask',mask)
    #cv2.imshow('res',res)
    return mask

def calculaHistogramaColor(colorImage):
    color = ('b','g','r')
    for i,col in enumerate(color):
        print("passei")
        histr = cv2.calcHist(colorImage,[i],None,[255],[0,255])
        plt.plot(histr,color = col)
        plt.xlim([0,255])
    plt.show()


def calculaHistograma(greyImage):
    histr = cv2.calcHist(greyImage,[0],None,[255],[0,256])
    plt.plot(histr,)
    plt.xlim([0,255])

def histogramaNormalizadoColor(ColorImg):
    lab = cv2.cvtColor(ColorImg, cv2.COLOR_BGR2LAB)
    
    lab_planes = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))

    lab_planes[0] = clahe.apply(lab_planes[0])

    lab = cv2.merge(lab_planes)

    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return bgr


#Contrast Limited Adaptive Histogram Equalization
# create a CLAHE object (Arguments are optional).  
def histogramaNormalizadoCinza(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(image)
    return cl1



def calculakmeans(img):
    z = img.reshape((-1,3))

    z = np.float32(z)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K=15

    ret, label,center = cv2.kmeans(z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)
    res = center[label.flatten()]
    res2= res.reshape((img.shape))

    return res2










if __name__ == "__main__":
    main()



