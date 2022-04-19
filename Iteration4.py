# Combining the color filters with lots of post processing

import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    width = int(cap.get(3))
    height = int(cap.get(4))

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_pink = np.array([150, 50, 150])
    upper_pink = np.array([170, 250, 255])

    lower_blue = np.array([80, 50, 160])
    upper_blue = np.array([95, 255, 250])

    lower_green = np.array([50, 50, 100])
    upper_green = np.array([90, 150, 250])

    lower_orange = np.array([1, 105, 150])
    upper_orange = np.array([90, 205, 255])

    lower_yellow = np.array([1, 100, 140])
    upper_yellow = np.array([50, 250, 250])

    mask = cv2.inRange(hsv, lower_pink, upper_pink)
    mask2 = cv2.inRange(hsv, lower_blue, upper_blue)
    mask3 = cv2.inRange(hsv, lower_green, upper_green)
    mask4 = cv2.inRange(hsv, lower_orange, upper_orange)
    mask5 = cv2.inRange(hsv, lower_yellow, upper_yellow)

    result = cv2.bitwise_and(frame, frame, mask=mask)
    result2 = cv2.bitwise_and(frame, frame, mask=mask2)
    result3 = cv2.bitwise_and(frame, frame, mask=mask3)
    result4 = cv2.bitwise_and(frame, frame, mask=mask4)
    result5 = cv2.bitwise_and(frame, frame, mask=mask5)

    result6 = cv2.bitwise_or(result, result2)
    result7 = cv2.bitwise_or(result3, result4)
    result8 = cv2.bitwise_or(result6, result5)
    result9 = cv2.bitwise_or(result7, result8)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
