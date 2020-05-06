import os
import cv2
import numpy as np

img_dir = 'F:/filtered_SSH/test/'
output_dir = 'F:/filtered_SSH_GRAY/test/'
num = 0
for img in os.listdir(img_dir):
    num += 1
    img_path = os.path.join(img_dir, img)
    img_bgr = cv2.imread(img_path)
    img_GRAY = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2GRAY)
    print(img_GRAY.shape)
    cv2.imwrite(output_dir + img, img_GRAY)
    if num == 730:
        break
