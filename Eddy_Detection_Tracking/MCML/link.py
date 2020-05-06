import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('G:/temporary/201610_04_origin.png', 0)
_, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
cv2.medianBlur(img, ksize=5)
image = cv2.Canny(img, threshold1=60, threshold2=90)
_, contours, hierarchy = cv2.findContours(thresh, 3, 2)
image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
img1 = cv2.imread('G:/temporary/201610_05_origin.png', 0)
_, thresh1 = cv2.threshold(img1, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
cv2.medianBlur(img1, ksize=5)
image1 = cv2.Canny(img1, threshold1=60, threshold2=90)
_, contours1, hierarchy1 = cv2.findContours(thresh1, 3, 2)
image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
coordinate = []
coordinate1 = []
for i in contours:
    cnt = i
    # cv2.drawContours(image, [cnt], 0, (0, 255, 0), 2)
    M = cv2.moments(cnt)
    # print(M)
    cX = int(M['m10'] / M['m00'])
    cY = int(M['m01'] / M['m00'])
    print('(%0.0f, %0.0f)' % (cX, cY))
    coordinate.append((cX, cY))
    cv2.circle(image, (cX, cY), 3, (255, 0, 0), -1, lineType=8, shift=0)
    # cv2.arrowedLine(image, (cX, cY), (cX1, cY1), (0, 0, 255), 1, 8, tipLength=0.1)
    # cv2.putText(image, '({},{})'.format(cX, cY), (cX - 20, cY - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
    # 最小外接圆及其半径
    # (x, y), radius = cv2.minEnclosingCircle(cnt)
    # (x, y, radius) = np.int0((x, y, radius))  # 圆心和半径取整
    # print(radius)
    # print((x, y, radius))
    # cv2.circle(image1, (x, y), radius, (0, 0, 255), 1)
    # x, y, w, h = cv2.boundingRect(cnt) #外接矩形
    # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2) #画出外接矩形
    # rect = cv2.minAreaRect(cnt) # 最小外接矩形
    # box = np.int0(cv2.boxPoints(rect))
    # cv2.drawContours(image, [box], 0, (0, 255, 0), 2)
# print(len(count_num))
for j in contours1:
    cnt = j
    # cv2.drawContours(image, [cnt], 0, (0, 255, 0), 2)
    M1 = cv2.moments(cnt)
    # print(M)
    cX1 = int(M1['m10'] / M1['m00'])
    cY1 = int(M1['m01'] / M1['m00'])
    print('(%0.0f, %0.0f)' % (cX1, cY1))
    coordinate1.append((cX1, cY1))
    cv2.circle(image1, (cX1, cY1), 3, (255, 0, 0), -1, lineType=8, shift=0)
    # cv2.arrowedLine(image, (cX, cY), (cX1, cY1), (0, 0, 255), 1, 8, tipLength=0.1)
    # cv2.putText(image, '({},{})'.format(cX, cY), (cX - 20, cY - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
    # 最小外接圆及其半径
    # (x, y), radius = cv2.minEnclosingCircle(cnt)
    # (x, y, radius) = np.int0((x, y, radius))  # 圆心和半径取整
    # print(radius)
    # print((x, y, radius))
    # cv2.circle(image1, (x, y), radius, (0, 0, 255), 1)
    # x, y, w, h = cv2.boundingRect(cnt) #外接矩形
    # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2) #画出外接矩形
    # rect = cv2.minAreaRect(cnt) # 最小外接矩形
    # box = np.int0(cv2.boxPoints(rect))
    # cv2.drawContours(image, [box], 0, (0, 255, 0), 2)
for k in range(len(coordinate)):
    if (coordinate1[k][0] in range(coordinate[k][0]-2, coordinate[k][0]+2+1)) and  \
            (coordinate1[k][1] in range(coordinate[k][1]-2, coordinate[k][1]+2+1)):
        cv2.arrowedLine(image, coordinate[k], coordinate1[k], (0, 0, 255), 1, 8, tipLength=0.1)
    else:
        cv2.arrowedLine(image, coordinate[k], coordinate[k], (0, 0, 255), 1, 8, tipLength=0.1)
cv2.imwrite('G:/temporary/test.png', image)

# exit()
# cv2.imshow('contours', image)
# cv2.waitKey(0)

