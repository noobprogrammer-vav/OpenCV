import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector()
classifier = Classifier("Model\keras_model.h5","Model\labels.txt")
offset = 10
imgsize = 400

#folder = "HandCricket\DataImages\C"
counter = 0
labels = ["One", "Two", "Three", "Four", "Five", "Six"]

while(cap.isOpened()):
    ret, img = cap.read()
    imgout = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x,y,w,h = hand['bbox']

        imgWhite = np.ones((imgsize,imgsize,3),np.uint8)*255
        imgcrop = img[ y-offset : y+h+offset , x-offset : x+w+offset ]

        

        aspectRatio = h/w

        if aspectRatio > 1:
            k = imgsize/h
            wcal = math.ceil(k*w)
            imgresize = cv2.resize(imgcrop,(wcal,imgsize))
            wgap = math.ceil((imgsize-wcal)/2)
            imgWhite[:, wgap:wcal+wgap] = imgresize
            prediction , index = classifier.getPrediction(imgWhite)
            print(prediction. index)
        
        else:
            k = imgsize/w
            hcal = math.ceil(k*h)
            imgresize = cv2.resize(imgcrop,(hcal,imgsize))
            hgap = math.ceil((imgsize-hcal)/2)
            imgWhite[:, hgap:hcal+hgap] = imgresize
            prediction , index = classifier.getPrediction(imgWhite)

        
        cv2.putText(imgout,labels[index],(x,y-20),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,255),2)
        cv2.rectangle(imgout,(x,y),(x+w,y+h),(255,0,255),4)
        #cv2.imshow("Croped",imgcrop)
        cv2.imshow("White",imgWhite)
    cv2.imshow("Image",imgout)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break