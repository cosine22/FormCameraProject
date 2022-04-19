# Code for finding multiple colored pixels at a time

import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    width = int(cap.get(3))
    height = int(cap.get(4))

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_pink = np.array([100, 70, 150])
    upper_pink = np.array([170, 120, 255])
    lower_blue = np.array([70,50,50])
    upper_blue = np.array([130,255,255])

    mask = cv2.inRange(hsv, lower_pink, upper_pink)
    mask2 = cv2.inRange(hsv, lower_blue, upper_blue)

    result = cv2.bitwise_and(frame, frame, mask=mask)
    avg = []
    print(result)
    for pixel in range(len(result)):
        if result[pixel][1] == [0,0,0]:
            continue
        else:
            avg.append(result[pixel])
    avgx = 0
    xvgy = 0
    for pixel1 in range(len(avg)):
        avgx += int(avg[pixel1][0])
        avgy += int(avg[pixel1][0])
    print(avgx)
    print(avgy)
    result2 = cv2.bitwise_and(frame, frame, mask=mask2)
    result3 = cv2.bitwise_or(result,result2)
    
    cv2.imshow('frame', result3)
    
    if cv2.waitKey(1) == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
