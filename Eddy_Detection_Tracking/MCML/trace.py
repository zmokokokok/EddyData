import cv2
import numpy as np
import matplotlib.pyplot as plt

img1 = cv2.imread('G:/temporary/201610_03_origin.png', 0)
_, thresh1 = cv2.threshold(img1, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
cv2.medianBlur(img1, ksize=5)
image1 = cv2.Canny(img1, threshold1=60, threshold2=90)
_, contours1, hierarchy1 = cv2.findContours(thresh1, 3, 2)
image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)

img2 = cv2.imread('G:/temporary/201610_04_origin.png', 0)
_, thresh2 = cv2.threshold(img2, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
cv2.medianBlur(img2, ksize=5)
image2 = cv2.Canny(img2, threshold1=60, threshold2=90)
_, contours2, hierarchy2 = cv2.findContours(thresh2, 3, 2)
image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)

img3 = cv2.imread('G:/temporary/201610_05_origin.png', 0)
_, thresh3 = cv2.threshold(img3, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
cv2.medianBlur(img3, ksize=5)
image3 = cv2.Canny(img3, threshold1=60, threshold2=90)
_, contours3, hierarchy3 = cv2.findContours(thresh3, 3, 2)
image3 = cv2.cvtColor(image3, cv2.COLOR_GRAY2BGR)

img4 = cv2.imread('G:/temporary/201610_06_origin.png', 0)
_, thresh4 = cv2.threshold(img4, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
cv2.medianBlur(img4, ksize=5)
image4 = cv2.Canny(img4, threshold1=60, threshold2=90)
_, contours4, hierarchy4 = cv2.findContours(thresh4, 3, 2)
image4 = cv2.cvtColor(image4, cv2.COLOR_GRAY2BGR)

img5 = cv2.imread('G:/temporary/201610_07_origin.png', 0)
_, thresh5 = cv2.threshold(img5, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
cv2.medianBlur(img5, ksize=5)
image5 = cv2.Canny(img5, threshold1=60, threshold2=90)
_, contours5, hierarchy5 = cv2.findContours(thresh5, 3, 2)
image5 = cv2.cvtColor(image5, cv2.COLOR_GRAY2BGR)
coordinate1 = []
coordinate2 = []
coordinate3 = []
coordinate4 = []
coordinate5 = []
for i in contours1:
    cnt = i
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
# print(len(count_num))
for j in contours2:
    cnt = j
    # cv2.drawContours(image, [cnt], 0, (0, 255, 0), 2)
    M2 = cv2.moments(cnt)
    # print(M)
    cX2 = int(M2['m10'] / M2['m00'])
    cY2 = int(M2['m01'] / M2['m00'])
    print('(%0.0f, %0.0f)' % (cX2, cY2))
    coordinate2.append((cX2, cY2))
    cv2.circle(image2, (cX2, cY2), 3, (255, 0, 0), -1, lineType=8, shift=0)
for m in contours3:
    cnt = m
    # cv2.drawContours(image, [cnt], 0, (0, 255, 0), 2)
    M3 = cv2.moments(cnt)
    # print(M)
    cX3 = int(M3['m10'] / M3['m00'])
    cY3 = int(M3['m01'] / M3['m00'])
    print('(%0.0f, %0.0f)' % (cX3, cY3))
    coordinate3.append((cX3, cY3))
    cv2.circle(image3, (cX3, cY3), 3, (255, 0, 0), -1, lineType=8, shift=0)
for n in contours4:
    cnt = n
    # cv2.drawContours(image, [cnt], 0, (0, 255, 0), 2)
    M4 = cv2.moments(cnt)
    # print(M)
    cX4 = int(M4['m10'] / M4['m00'])
    cY4 = int(M4['m01'] / M4['m00'])
    print('(%0.0f, %0.0f)' % (cX4, cY4))
    coordinate4.append((cX4, cY4))
    cv2.circle(image4, (cX4, cY4), 3, (255, 0, 0), -1, lineType=8, shift=0)
for p in contours5:
    cnt = p
    # cv2.drawContours(image, [cnt], 0, (0, 255, 0), 2)
    M5 = cv2.moments(cnt)
    # print(M)
    cX5 = int(M5['m10'] / M5['m00'])
    cY5 = int(M5['m01'] / M5['m00'])
    print('(%0.0f, %0.0f)' % (cX5, cY5))
    coordinate5.append((cX5, cY5))
    cv2.circle(image5, (cX5, cY5), 3, (255, 0, 0), -1, lineType=8, shift=0)
for k in range(len(coordinate1)):
    if (coordinate2[k][0] in range(coordinate1[k][0]-2, coordinate1[k][0]+2+1)) and  \
            (coordinate2[k][1] in range(coordinate1[k][1]-2, coordinate1[k][1]+2+1)):
        cv2.arrowedLine(image1, coordinate1[k], coordinate2[k], (0, 0, 255), 1, 8, tipLength=0.1)
    else:
        cv2.arrowedLine(image1, coordinate1[k], coordinate1[k], (0, 0, 255), 1, 8, tipLength=0.1)
cv2.imwrite('G:/temporary/trace.png', image1)

# exit()
# cv2.imshow('contours', image)
# cv2.waitKey(0)

