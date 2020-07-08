# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 00:34:31 2020

@author: Kushani Ranaweera
"""


import cv2
import numpy as np


cap = cv2.VideoCapture("C:/Users/Kushani Ranaweera/Documents/tharka/6th semester/CO543/project/social-distance-detector/case3_2.mp4")



while True:
    (grabbed,frame)=cap.read()
    (grabbed,frame2)=cap.read()  
    
    if not grabbed:
        break
        

    video = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    blur = cv2.GaussianBlur(frame,(3,3),0)

    sobelx = cv2.Sobel(blur,cv2.CV_64F,1,0,ksize=3)  # x
    sobely = cv2.Sobel(blur,cv2.CV_64F,0,1,ksize=5)  # y
    laplacian = cv2.Laplacian(blur,cv2.CV_64F)
    median = cv2.medianBlur(blur,5)
    smooth = cv2.addWeighted( blur, 1.2, frame, -0.1, 0)
    
    
    
    dst = cv2.bilateralFilter(smooth, 0, 40, 10)



    #cv2.imshow("Frame", frame)
    #cv2.imshow("blur", blur)
    #cv2.imshow("Smooth", smooth)
    cv2.imshow("restored video", dst)    
    key = cv2.waitKey(1) & 0xFF

	# if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break
        
        
    
# close all windows
cv2.destroyAllWindows()