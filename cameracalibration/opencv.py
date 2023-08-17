import cv2
import numpy
cap=cv2.VideoCapture(1)
cap.set(6, cv2.VideoWriter.fourcc('M','J','P','G'))
while(True):
    ret,frame=cap.read()
    cv2.imshow("video",frame)
    if cv2.waitKey(10)==ord("q"):
        break