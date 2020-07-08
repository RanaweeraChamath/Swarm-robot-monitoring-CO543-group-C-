# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 11:20:18 2020

@author: Kushani Ranaweera
"""

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
cap = cv.VideoCapture('C:/Users/Kushani Ranaweera/Documents/tharka/6th semester/CO543/project/Implenentations/case3_2.mp4')
# create a list of first 5 frames

while True:
    (grabbed, im) = cap.read()
    
    #frame = imutils.resize(frame, width=700)
    for i in range(2,im.size[0]-2):
        for j in range(2,im.size[1]-2):
            b=[]
            if im.getpixel((i,j))>0 and im.getpixel((i,j))<255:
                pass
            elif im.getpixel((i,j))==0 or im.getpixel((i,j))==255:
                c=0
                for p in range(i-1,i+2):
                    for q in range(j-1,j+2):
                        if im.getpixel((p,q))==0 or im.getpixel((p,q))==255: 
                            c=c+1
                    if c>6:
                        c=0
                        for p in range(i-2,i+3):
                            for q in range(j-2,j+3):
                                b.append(im.getpixel((p,q)))
                                if im.getpixel((p,q))==0 or im.getpixel((p,q))==255:
                                    c=c+1
                        if c==25:
                            a=sum(b)/25
                        #print a
                            im.putpixel((i,j),a)
                        else:
                            p=[]
                            for t in b:
                                if t not in (0,255):
                                    p.append(t)
                            p.sort()
                            im.putpixel((i,j),p[len(p)/2])
                    else:
                        b1=[]
                        for p in range(i-1,i+2):
                            for q in range(j-1,j+2):
                                b1.append(im.getpixel((p,q)))
                        im.putpixel((i,j),sum(b1)/9)

    cv.imshow("Frame", im)
    key = cv.waitKey(1) & 0xFF

	# if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break
	   
    
# close all windows
cv.destroyAllWindows()
	
	

	# if the 'q' key is pressed, stop the loop
	
		
    
# convert all to grayscale

# convert all to float64

# create a noise of variance 25

# Add this noise to images

# Convert back to uint8

# Denoise 3rd frame considering all the 5 frames

#
