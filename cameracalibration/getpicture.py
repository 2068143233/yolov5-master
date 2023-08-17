import cv2
cap=cv2.VideoCapture(1)
count=0
count2=0
while(True):
    success,frame=cap.read()
    cv2.imshow("video",frame)
    cv2.waitKey(1)
    if count%60==0 and success :
        count2=count2+1
        cv2.imwrite('new'+str(count2)+'.jpg',frame)
    count=count+1