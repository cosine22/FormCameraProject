#Saves a video and then does post video analysis of the pixels

import numpy as np
import cv2

cap = cv2.VideoCapture(0)
x = 0
frameavg = []
width = int(cap.get(3))
height = int(cap.get(4))
size = (width, height)

video = cv2.VideoWriter('filename.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size)

while x < 50:
    ret, frame = cap.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_pink = np.array([100, 70, 150])
    upper_pink = np.array([170, 120, 255])

    mask = cv2.inRange(hsv, lower_pink, upper_pink)
    result = cv2.bitwise_and(frame, frame, mask=mask)
    video.write(result)

    if cv2.waitKey(100) == ord('q'):
        break
    x += 1

video.release()
cap.release()
rows, cols, _ = result.shape
cap = cv2.VideoCapture('filename.avi')
for l in range(x):
    ret, frame = cap.read()
    avg = []
    for i in range(rows):
        for j in range(cols):
            if (frame[i][j][0] > 0):
                avg.append(frame[i, j])

    avgx = 0
    avgy = 0
    for pixel1 in range(len(avg)):
        avgx += int(avg[pixel1][0])
        avgy += int(avg[pixel1][1])
    avgx = avgx / int(len(avg))
    avgy = avgy / int(len(avg))
    frameavg.append([avgx, avgy])

cv2.destroyAllWindows()
cap.release()
