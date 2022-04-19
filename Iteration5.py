#After image processing and pixel averages calculated, this performs
# cartesian calculations to find where colors are in relation to each other
# and assigns these colors to different joints to give advice

import numpy as np
import cv2

cap = cv2.VideoCapture(0)
x = 0
frameavgpink = []
frameavgorange = []
frameavgblue = []
frameavggreen = []
frameavgyellow = []
frameavg = [frameavgpink, frameavgblue, frameavggreen, frameavgorange, frameavgyellow]

width = int(cap.get(3))
height = int(cap.get(4))
size = (width, height)
lower_pink = np.array([160, 70, 200])
upper_pink = np.array([190, 200, 255])

lower_blue = np.array([65, 70, 70])
upper_blue = np.array([80, 200, 200])

lower_green = np.array([35, 70, 70])
upper_green = np.array([64, 150, 255])

lower_orange = np.array([1, 170, 240])
upper_orange = np.array([17, 235, 255])

lower_yellow = np.array([20, 90, 125])
upper_yellow = np.array([30, 255, 250])

video = cv2.VideoWriter('filename.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size)

while x < 50:
    ret, frame = cap.read()

    video.write(frame)

    if cv2.waitKey(100) == ord('q'):
        break
    x += 1

video.release()
cap.release()
cv2.destroyAllWindows()
rows, cols, _ = frame.shape
cap = cv2.VideoCapture('filename.avi')

for l in range(x):
    ret, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_pink, upper_pink)
    mask1 = cv2.inRange(hsv, lower_blue, upper_blue)
    mask2 = cv2.inRange(hsv, lower_green, upper_green)
    mask3 = cv2.inRange(hsv, lower_orange, upper_orange)
    mask4 = cv2.inRange(hsv, lower_yellow, upper_yellow)

    result = cv2.bitwise_and(frame, frame, mask=mask)
    result1 = cv2.bitwise_and(frame, frame, mask=mask1)
    result2 = cv2.bitwise_and(frame, frame, mask=mask2)
    result3 = cv2.bitwise_and(frame, frame, mask=mask3)
    result4 = cv2.bitwise_and(frame, frame, mask=mask4)

    results = [result, result1, result2, result3, result4]
    y = 0
    for result in results:
        y += 1

        avg = []
        for i in range(rows):
            for j in range(cols):
                if (result[i][j][0] > 0):
                    avg.append([i, j])
        avgx = 0
        avgy = 0
        for pixel1 in range(len(avg)):
            avgx += int(avg[pixel1][0])
            avgy += int(avg[pixel1][1])
        try:
            avgx = avgx / int(len(avg))
        except ZeroDivisionError:
            if y == 1:
                avgx = frameavg[0][-1][0]
            elif y == 2:
                avgx = frameavg[1][-1][0]
            elif y == 3:
                avgx = frameavg[2][-1][0]
            elif y == 4:
                avgx = frameavg[3][-1][0]
            elif y == 5:
                avgx = frameavg[4][-1][0]

        try:
            avgy = avgy / int(len(avg))
        except ZeroDivisionError:
            if y == 1:
                avgy = frameavg[0][-1][1]
            elif y == 2:
                avgy = frameavg[1][-1][1]
            elif y == 3:
                avgy = frameavg[2][-1][1]
            elif y == 4:
                avgy = frameavg[3][-1][1]
            elif y == 5:
                avgy = frameavg[4][-1][1]

        if y == 1:
            frameavg[0].append([int(avgx), int(avgy)])
        elif y == 2:
            frameavg[1].append([int(avgx), int(avgy)])
        elif y == 3:
            frameavg[2].append([int(avgx), int(avgy)])
        elif y == 4:
            frameavg[3].append([int(avgx), int(avgy)])
        elif y == 5:
            frameavg[4].append([int(avgx), int(avgy)])
cv2.destroyAllWindows()
cap.release()
cap = cv2.VideoCapture('filename.avi')

video = cv2.VideoWriter('filename1.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size)
for a in range(x):
    ret, frame = cap.read()
    cv2.circle(frame, (frameavg[0][a][1], frameavg[0][a][0]), 15, (0, 0, 255), 5)
    cv2.circle(frame, (frameavg[1][a][1], frameavg[1][a][0]), 15, (255, 0, 0), 5)
    cv2.circle(frame, (frameavg[2][a][1], frameavg[2][a][0]), 15, (0, 255, 0), 5)
    cv2.circle(frame, (frameavg[3][a][1], frameavg[3][a][0]), 15, (255, 255, 255), 5)
    cv2.circle(frame, (frameavg[4][a][1], frameavg[4][a][0]), 15, (0, 0, 0), 5)
    video.write(frame)
    cv2.imshow('frame', frame)
video.release()
cap.release()


def replay():
    while True:
        cap = cv2.VideoCapture('filename.avi')
        for a in range(x):
            ret, frame = cap.read()
            cv2.circle(frame, (frameavg[0][a][1], frameavg[0][a][0]), 15, (0, 0, 255), 5)
            cv2.circle(frame, (frameavg[1][a][1], frameavg[1][a][0]), 15, (255, 0, 0), 5)
            cv2.circle(frame, (frameavg[2][a][1], frameavg[2][a][0]), 15, (0, 255, 0), 5)
            cv2.circle(frame, (frameavg[3][a][1], frameavg[3][a][0]), 15, (255, 255, 255), 5)
            cv2.circle(frame, (frameavg[4][a][1], frameavg[4][a][0]), 15, (0, 0, 0), 5)
            cv2.imshow('frame', frame)
            if cv2.waitKey(100) == ord('q'):
                break
        if cv2.waitKey(10000) == ord('q'):
            break


replay()
cv2.destroyAllWindows()
