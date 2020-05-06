# import numpy as np
#
# a = np.zeros((3, 4, 4))
# a[0, 1, 1] = 255
# a[2, 1, 2] = 255
# a = (a > 0).astype(np.float)
# b = np.array([1, 0, 2]).reshape((3, 1, 1))
# c = (a * b).sum(axis=0)
# print(c)










import numpy as np
import cv2
import os

training_images = []
input_data = 'F:/Fill_Hole/train/'
output_data = 'F:/groundtruth/train_groundtruth_Segmentation.npy'

for image in os.listdir(input_data):
    img_path = os.path.join(input_data, image)
    im_in_rgb = cv2.imread(img_path).astype(np.uint32)
    img=im_in_rgb
    for h in range(img.shape[0]):
        for w in range(img.shape[1]):
            if img[h, w, 0] == 255 and img[h, w, 1] == 255 and img[h, w, 2] == 255:
                img[h, w, :] = [0, 0, 0]
                continue
            colors = img[h, w, :]
            colors = np.argmax(colors)
            if colors == 0:
                img[h, w, :] = [0, 255, 255]
            elif colors == 2:
                img[h, w, :] = [255, 255, 0]
    im_in_rgb=img
    a = (im_in_rgb > 0).astype(np.int)
    b = np.array([1, 0, 2]).reshape((1, 1, 3))
    c = (a * b).sum(axis=2)
    training_images.append(c)
    np.save(output_data, training_images)
