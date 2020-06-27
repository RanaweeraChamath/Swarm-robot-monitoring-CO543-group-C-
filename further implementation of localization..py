# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 11:41:34 2020

@author: Kushani Ranaweera
"""
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
	help="max buffer size")
ap.add_argument("-o", "--output", type=str, default="",
	help="path to (optional) output video file")
ap.add_argument("-y", "--yolo", type=str, default="",
	help="base path to YOLO directory")
args = vars(ap.parse_args())

writer = None
vs = cv2.VideoCapture('C:/Users/Kushani Ranaweera/Documents/tharka/6th semester/CO543/project/Implenentations/case3_2.mp4')


Lower = (110, 50, 50)
Upper = (130, 255, 255)
pts = deque(maxlen=args["buffer"])
while True:
    (grabbed, frame1)=vs.read()
    (grabbed, frame2)=vs.read()   

    frame1 = imutils.resize(frame1, width=700)
    blurred = cv2.GaussianBlur(frame1, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

	# construct a mask for the swarm robots, then perform
	# a series of dilations and erosions to remove any small
	# blobs left in the mask
    mask = cv2.inRange(hsv, Lower, Upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)    
    
    
    edges = cv2.Canny(frame1, 100,200)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    coordinates = []
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None
    #  drawing a rectangle around detected movements and calculating the centers of the objects
    for contour in contours:
        if cv2.contourArea(contour) < 50 or cv2.contourArea(contour) > 5000:
            continue
        (x, y, w, h) = cv2.boundingRect(contour)
        coordinates = coordinates + [(int(x + w / 2), int(y + h / 2))]
        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 1)

    

    #  drawing a line among moving objects to give a visual representation of the distance
    if len(cnts) > 0:
	    c = max(cnts, key=cv2.contourArea)
	    ((x, y), radius) = cv2.minEnclosingCircle(c)
	    M = cv2.moments(c)
	    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

		# only proceed if the radius meets a minimum size
	    if radius > 10:
			# draw the circle and centroid on the frame,
			# then update the list of tracked points
		    cv2.circle(frame1, (int(x), int(y)), int(radius),
				(0, 255, 255), 2)
		    cv2.circle(frame1, center, 5, (0, 0, 255), -1)
    for i in range(1, len(pts)):
		# if either of the tracked points are None, ignore
		# them
	    if pts[i - 1] is None or pts[i] is None:
		    continue

		# otherwise, compute the thickness of the line and
		# draw the connecting lines
	    thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
	    cv2.line(frame1, pts[i - 1], pts[i], (0, 0, 255), thickness)


    

    if len(coordinates) > 0:
       for element in range(len(coordinates)):
            other = element + 1

            while other < len(coordinates):
                other = other + 1
                cv2.line(frame1, coordinates[element], coordinates[other - 1], (0, 255, 0), thickness=1, lineType=8)

	#out=cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc(*'MJPG'),10,(640,480))
	#out.write(frame)
	#out.release()
    # show the frame to our screen
    if args["output"] != "" and writer is None:
		# initialize our video writer
	    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
	    writer = cv2.VideoWriter(args["output"], fourcc, 25,
			(frame1.shape[1], frame1.shape[0]), True)
    
    cv2.imshow("Frame", frame1)


    if cv2.waitKey(5) & 0xFF == ord('q'):
        break    

cv2.destroyAllWindows()