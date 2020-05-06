import cv2
import numpy as np
import os

# img_dir = 'F:/PET_result/train/'
# output_dir = 'F:/Fill_Hole/train/'
img_dir = 'F:/temporary/fill_input/'
output_dir = 'F:/temporary/fill_output/'
def FillHole_RGB(imgPath, SavePath):

    im_in_rgb = cv2.imread(imgPath).astype(np.uint32)
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
    im_in_rgb= 255-im_in_rgb
    # 将im_in_rgb的RGB颜色转换为 0xbbggrr
    im_in_lbl = im_in_rgb[:, :, 0] + (im_in_rgb[:, :, 1] << 8) + (im_in_rgb[:, :, 2] << 16)

    # 将0xbbggrr颜色转换为0,1,2,...
    colors, im_in_lbl_new = np.unique(im_in_lbl, return_inverse=True)

    # 将im_in_lbl_new数组reshape为2维
    im_in_lbl_new = np.reshape(im_in_lbl_new, im_in_lbl.shape)

    # 创建从32位im_in_lbl_new到8位colorize颜色的映射
    colorize = np.empty((len(colors), 3), np.uint8)
    colorize[:, 0] = (colors & 0x0000FF)
    colorize[:, 1] = (colors & 0x00FF00) >> 8
    colorize[:, 2] = (colors & 0xFF0000) >> 16

    # 输出一下colorize中的color
    print("Colors_RGB: \n", colorize)

    # 有几种颜色就设置几层数组，每层数组均为各种颜色的二值化数组
    im_result = np.zeros((len(colors),) + im_in_lbl_new.shape, np.uint8)

    # 初始化二值数组
    im_th = np.zeros(im_in_lbl_new.shape, np.uint8)

    for i in range(len(colors)):
        for j in range(im_th.shape[0]):
            for k in range(im_th.shape[1]):
                if (im_in_lbl_new[j][k] == i):
                    im_th[j][k] = 255
                else:
                    im_th[j][k] = 0
        # 复制 im_in 图像
        im_floodfill = im_th.copy()

        # Mask 用于 floodFill,mask多出来的2可以保证扫描的边界上的像素都会被处理.
        h, w = im_th.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)

        isbreak = False
        for m in range(im_floodfill.shape[0]):
            for n in range(im_floodfill.shape[1]):
                if (im_floodfill[m][n] == 0):
                    seedPoint = (m, n)
                    isbreak = True
                    break
            if (isbreak):
                break
        # 得到im_floodfill
        cv2.floodFill(im_floodfill, mask, seedPoint, 255, 4)

        #        help(cv2.floodFill.rect)
        # 得到im_floodfill的逆im_floodfill_inv
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)

        # 把im_in、im_floodfill_inv这两幅图像结合起来得到前景
        im_out = im_th | im_floodfill_inv
        im_result[i] = im_out

    # rgb结果图像
    im_fillhole = np.zeros((im_in_lbl_new.shape[0], im_in_lbl_new.shape[1], 3), np.uint8)

    # 之前的颜色映射起到了作用
    for i in range(im_result.shape[1]):
        for j in range(im_result.shape[2]):
            for k in range(im_result.shape[0]):
                if (im_result[k][i][j] == 255):
                    im_fillhole[i][j] = colorize[k]
                    break

    # 保存图像
    cv2.imwrite(SavePath, im_fillhole)

def FillHole_RGB_(imgPath, SavePath):
    # 读取图像为uint32,之所以选择uint32是因为下面转为0xbbggrr不溢出
    im_in_rgb = cv2.imread(imgPath).astype(np.uint32)
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
    im_in_rgb= 255-im_in_rgb
    # 将im_in_rgb的RGB颜色转换为 0xbbggrr
    im_in_lbl = im_in_rgb[:, :, 0] + (im_in_rgb[:, :, 1] << 8) + (im_in_rgb[:, :, 2] << 16)

    # 将0xbbggrr颜色转换为0,1,2,...
    colors, im_in_lbl_new = np.unique(im_in_lbl, return_inverse=True)

    # 将im_in_lbl_new数组reshape为2维
    im_in_lbl_new = np.reshape(im_in_lbl_new, im_in_lbl.shape)

    # 创建从32位im_in_lbl_new到8位colorize颜色的映射
    colorize = np.empty((len(colors), 3), np.uint8)
    colorize[:, 0] = (colors & 0x0000FF)
    colorize[:, 1] = (colors & 0x00FF00) >> 8
    colorize[:, 2] = (colors & 0xFF0000) >> 16

    # 输出一下colorize中的color
    print("Colors_RGB: \n", colorize)

    # 有几种颜色就设置几层数组，每层数组均为各种颜色的二值化数组
    im_result = np.zeros((len(colors),) + im_in_lbl_new.shape, np.uint8)

    # 初始化二值数组
    im_th = np.zeros(im_in_lbl_new.shape, np.uint8)

    for i in range(len(colors)):
        for j in range(im_th.shape[0]):
            for k in range(im_th.shape[1]):
                if (im_in_lbl_new[j][k] == i):
                    im_th[j][k] = 255
                else:
                    im_th[j][k] = 0
        # 复制 im_in 图像
        im_floodfill = im_th.copy()

        # Mask 用于 floodFill,mask多出来的2可以保证扫描的边界上的像素都会被处理.
        h, w = im_th.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)

        isbreak = False
        for m in range(im_floodfill.shape[0]):
            for n in range(im_floodfill.shape[1]):
                if (im_floodfill[m][n] == 0):
                    seedPoint = (n, m)
                    isbreak = True
                    break
            if (isbreak):
                break
        # 得到im_floodfill
        cv2.floodFill(im_floodfill, mask, seedPoint, 255, 4)

        #        help(cv2.floodFill.rect)
        # 得到im_floodfill的逆im_floodfill_inv
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)

        # 把im_in、im_floodfill_inv这两幅图像结合起来得到前景
        im_out = im_th | im_floodfill_inv
        im_result[i] = im_out

    # rgb结果图像
    im_fillhole = np.zeros((im_in_lbl_new.shape[0], im_in_lbl_new.shape[1], 3), np.uint8)

    # 之前的颜色映射起到了作用
    for i in range(im_result.shape[1]):
        for j in range(im_result.shape[2]):
            for k in range(im_result.shape[0]):
                if (im_result[k][i][j] == 255):
                    im_fillhole[i][j] = colorize[k]
                    break

    # 保存图像
    cv2.imwrite(SavePath, im_fillhole)

num = 0
for img in os.listdir(img_dir):
    num += 1
    img_path = os.path.join(img_dir, img)
    out_item_name = img.split('.')[0] + '.png'
    try:
        FillHole_RGB(img_path, output_dir + out_item_name)
    except:
        FillHole_RGB_(img_path, output_dir + out_item_name)
    if num == 4750:
        break
