import cv2
import numpy as np

img = cv2.imread('G:/temporary/20161003origin.png', 0)
_, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
cv2.medianBlur(img, ksize=5)
image = cv2.Canny(img, threshold1=60, threshold2=90)
_, contours, hierarchy = cv2.findContours(thresh, 3, 2)
image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
image1 = np.copy(image)
def convert_RGB_(imgPath, SavePath):
    im_in_rgb = cv2.imread(imgPath).astype(np.uint32)
    img = im_in_rgb
    for h in range(img.shape[0]):
        for w in range(img.shape[1]):
            if img[h, w, 0] == 255 and img[h, w, 1] == 255 and img[h, w, 2] == 255:
                img[h, w, :] =[0, 0, 0]
                continue
            colors = img[h, w, :]
            colors = np.argmax(colors)
            if colors == 0:
                img[h, w, :] = [0, 255, 255]
            elif colors == 2:
                img[h, w, :] = [255, 255, 0]
    im_in_rgb = img
    im_in_rgb = 255 - im_in_rgb
    return im_in_rgb
for i in contours:
    cnt = i
    # cv2.drawContours(image, [cnt], 0, (0, 255, 0), 2)
    M = cv2.moments(cnt)
    # print(M)
    cX = int(M['m10'] / M['m00'])
    cY = int(M['m01'] / M['m00'])
    print('(%0.0f, %0.0f)' % (cX, cY))
    cv2.circle(image, (cX, cY), 3, (255, 0, 0), -1, lineType=8, shift=0)
    cv2.putText(image, '({},{})'.format(cX, cY), (cX - 20, cY - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
    # 最小外接圆及其半径
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    (x, y, radius) = np.int0((x, y, radius))  # 圆心和半径取整
    print(radius)
    print((x, y, radius))
    cv2.circle(image1, (x, y), radius, (0, 0, 255), 1)
    # x, y, w, h = cv2.boundingRect(cnt) #外接矩形
    # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2) #画出外接矩形
    # rect = cv2.minAreaRect(cnt) # 最小外接矩形
    # box = np.int0(cv2.boxPoints(rect))
    # cv2.drawContours(image, [box], 0, (0, 255, 0), 2)
cv2.imwrite('G:/temporary/20161003_circle_.png', image1)

# exit()
# cv2.imshow('contours', image)
# cv2.waitKey(0)

