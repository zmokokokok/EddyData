import numpy as np
input_data = 'F:/filtered_SSH_GRAY/train/'
output_data = 'F:/groundtruth/filtered_SSH_gray_train_data.npy'
import os
import cv2


training_images = []

for img in os.listdir(input_data):
    img_path = os.path.join(input_data, img)
    a = cv2.imread(img_path)
    c = a[:,:,0]
    b = np.array(c)
    training_images.append(b)
    np.save(output_data, training_images)

