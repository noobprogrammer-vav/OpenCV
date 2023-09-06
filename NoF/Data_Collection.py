import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector()
offset = 10
imgsize = 400

folder = f"D:\Adhi\Codings\py3\Computer_Vision\CV\YT2\Random Projects\HandCricket\DataImages\Four"
counter = 0

while(cap.isOpened()):
    ret, frame = cap.read()
    hands, img = detector.findHands(frame)
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
        
        else:
            k = imgsize/w
            hcal = math.ceil(k*h)
            imgresize = cv2.resize(imgcrop,(hcal,imgsize))
            hgap = math.ceil((imgsize-hcal)/2)
            imgWhite[:, hgap:hcal+hgap] = imgresize


        #cv2.imshow("Croped",imgcrop)
        cv2.imshow("White",imgWhite)
    cv2.imshow("Image",frame)

    if (cv2.waitKey(1) & 0xFF == ord('s')):
        if (counter >=400) and (counter<800):
            folder = f"D:\Adhi\Codings\py3\Computer_Vision\CV\YT2\Random Projects\HandCricket\DataImages\Five"
        elif (counter >= 800):
            folder = f"D:\Adhi\Codings\py3\Computer_Vision\CV\YT2\Random Projects\HandCricket\DataImages\Six"
        cv2.imwrite(f'{folder}/img_{time.time()}.jpg',imgWhite)
        counter +=1
        print(counter)
    elif cv2.waitKey(1) & 0xFF == ord('q'):
        break