# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 10:36:44 2020

@author: Kushani Ranaweera
"""
import numpy as np
import argparse
import cv2
import imutils
import time
from PIL import Image, ImageFilter

vs = cv2.VideoCapture('C:/Users/Kushani Ranaweera/Documents/tharka/6th semester/CO543/project/Implenentations/case3_2.mp4')

Lower = (110, 50, 50)
Upper = (130, 255, 255)


while True :
	# grab the current frame
	(grabbed, frame) = vs.read()
    

	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if not grabbed:
		break
    


	kernel=np.ones((15,15),np.float32)/255    
    
	#mb=cv2.medianBlur(frame, 3)
	frame = Image.fromarray(frame.astype('uint8'))    
	new_image = frame.filter(ImageFilter.UnsharpMask(radius=2, percent=150))   
    
    #new_image = image.filter(ImageFilter.UnsharpMask(radius=2, percent=150))
    #smoothed=cv2.filter2D(frame,-1,kernel)    
	#smoothed=cv2.filter2D(frame,-1,kernel)
	#blurred = cv2.GaussianBlur(smoothed, (15, 15), 0)      
    
    #new_image = image.filter(ImageFilter.UnsharpMask(radius=2, percent=150))
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break