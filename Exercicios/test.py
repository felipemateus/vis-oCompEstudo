import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import time
import copy
count_mvt = 0

size_box = 10

cap = cv.VideoCapture('lena_cor.jpg')
img = cv.imread("lena_cor.jpg")


move_box = 0
old_mox = copy.deepcopy(img[0:size_box , 0:size_box])

frame_rate = 2
prev = 0

if not cap.isOpened():
    print("Não foi possivel abrir")
    exit()
    
while True:
 
    time_elapsed = time.time() -prev

    if time_elapsed > 1./frame_rate:
        img[0:size_box , count_mvt:size_box+count_mvt] = move_box

        prev =time.time()
        gray = img
        cv.imshow('frame', gray)
        cv.imshow('old_max',old_mox)
        #cv.imshow('move_box',move_box)
        
        print("frame")
        #corrige o box

        #img[count_mvt:size_box+count_mvt , count_mvt:size_box+count_mvt] = old_mox
        #old_mox = img[count_mvt:size_box+count_mvt , count_mvt:size_box+count_mvt]
        #img[count_mvt:size_box+count_mvt , count_mvt:size_box+count_mvt] = move_box

        print("0:%d , %d:%d" %(size_box,count_mvt,size_box+count_mvt))
        
        img[0:size_box , count_mvt:size_box+count_mvt] = old_mox
        
        count_mvt +=1
        old_mox = copy.deepcopy(img[0:size_box , count_mvt:size_box+count_mvt])
        
        print(old_mox)



        




    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()