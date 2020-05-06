import cv2


# 腐蚀算法
import cv2
import numpy as np

img = cv2.imread('F:/temporary/20161003.png')
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
ret, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
# kernel = np.ones((4, 4),np.uint8)
# erosion = cv2.erode(thresh, kernel)

cv2.imwrite('F:/temporary/201610_03_origin.png', thresh)




# c++ 求质心并标注
'''
#include "stdafx.h"  
#include <iostream>  
#include <opencv2/core/core.hpp>  
#include <opencv/cv.hpp>  
#include <opencv2/highgui/highgui.hpp>  
   
using namespace cv;   
using namespace std;   
Mat src;   
Mat src_gray;   
int thresh = 30;   
int max_thresh = 255;   
   
int main()  
{     
    src = imread( "C:\\Users\\Lijunliang\\Desktop\\opencvpic\\2\\22.png" ,CV_LOAD_IMAGE_COLOR );    //注意路径得换成自己的  
    cvtColor( src, src_gray, CV_BGR2GRAY );//灰度化      
    GaussianBlur( src, src, Size(3,3), 0.1, 0, BORDER_DEFAULT );      
    blur( src_gray, src_gray, Size(3,3) ); //滤波       
    namedWindow( "image", CV_WINDOW_AUTOSIZE );       
    imshow( "image", src );       
    moveWindow("image",20,20);    
    //定义Canny边缘检测图像       
    Mat canny_output;     
    vector<vector<Point> > contours;      
    vector<Vec4i> hierarchy;    
    //利用canny算法检测边缘       
    Canny( src_gray, canny_output, thresh, thresh*3, 3 );     
    namedWindow( "canny", CV_WINDOW_AUTOSIZE );       
    imshow( "canny", canny_output );      
    moveWindow("canny",550,20);       
    //查找轮廓    
    findContours( canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );     
    //计算轮廓矩       
    vector<Moments> mu(contours.size() );       
    for( int i = 0; i < contours.size(); i++ )     
    {   
        mu[i] = moments( contours[i], false );   
    }     
    //计算轮廓的质心     
    vector<Point2f> mc( contours.size() );      
    for( int i = 0; i < contours.size(); i++ )     
    {   
        mc[i] = Point2d( mu[i].m10/mu[i].m00 , mu[i].m01/mu[i].m00 );   
    }     
    //画轮廓及其质心并显示      
    Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );         
    for( int i = 0; i< contours.size(); i++ )      
    {         
        Scalar color = Scalar( 255, 0, 0);        
        drawContours( drawing, contours, i, color, 2, 8, hierarchy, 0, Point() );         
        circle( drawing, mc[i], 5, Scalar( 0, 0, 255), -1, 8, 0 );                
        rectangle(drawing, boundingRect(contours.at(i)), cvScalar(0,255,0));              
        char tam[100];   
        sprintf(tam, "(%0.0f,%0.0f)",mc[i].x,mc[i].y);   
        putText(drawing, tam, Point(mc[i].x, mc[i].y), FONT_HERSHEY_SIMPLEX, 0.4, cvScalar(255,0,255),1);     
    }     
    namedWindow( "Contours", CV_WINDOW_AUTOSIZE );    
    imshow( "Contours", drawing );    
    moveWindow("Contours",1100,20);       
    waitKey(0);       
    src.release();    
    src_gray.release();       
    return 0;   
}
'''
# 分水岭算法
# import numpy as np
# import cv2
# from matplotlib import pyplot as plt
#
# # img = cv2.imread('F:/temporary/111.png')
# img = cv2.imread('./20160919.png')
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
#
#
# # noise removal
# kernel = np.ones((3,3),np.uint8)
# opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2) # 形态开运算
#
# # sure background area
# sure_bg = cv2.dilate(opening,kernel,iterations=3)
#
# # Finding sure foreground area
# dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
# ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
#
# # Finding unknown region
# sure_fg = np.uint8(sure_fg)
# unknown = cv2.subtract(sure_bg,sure_fg)
#
#
# # Marker labelling
# ret, markers = cv2.connectedComponents(sure_fg)
#
# # Add one to all labels so that sure background is not 0, but 1
# markers = markers+1
#
# # Now, mark the region of unknown with zero
# markers[unknown==255] = 0
#
# markers = cv2.watershed(img,markers)
# img[markers == -1] = [255,0,0]
#
#
# cv2.imshow('img',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

'''
import cv2
import numpy as np

# 载入手写数字图片
img = cv2.imread('handwriting.jpg', 0)
_, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
image, contours, hierarchy = cv2.findContours(thresh, 3, 2)

# 创建出两幅彩色图用于绘制
img_color1 = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
img_color2 = np.copy(img_color1)

# 以数字3的轮廓为例
cnt = contours[0]
cv2.drawContours(img_color1, [cnt], 0, (0, 0, 255), 2)

# 1.轮廓面积
area = cv2.contourArea(cnt)  # 4386.5
print(area)


# 2.轮廓周长
perimeter = cv2.arcLength(cnt, True)  # 585.7716
print(perimeter)


# 3.图像矩
M = cv2.moments(cnt)
print(M)
print(M['m00'])  # 同前面的面积：4386.5
cx, cy = M['m10'] / M['m00'], M['m01'] / M['m00']  # 质心
print(cx, cy)

# 4.图像外接矩形和最小外接矩形
x, y, w, h = cv2.boundingRect(cnt)  # 外接矩形
cv2.rectangle(img_color1, (x, y), (x + w, y + h), (0, 255, 0), 2)

rect = cv2.minAreaRect(cnt)  # 最小外接矩形
box = np.int0(cv2.boxPoints(rect))  # 矩形的四个角点并取整
# 也可以用astype(np.int)取整
cv2.drawContours(img_color1, [box], 0, (255, 0, 0), 2)

cv2.imshow('contours', img_color1)
cv2.waitKey(0)


# 5.最小外接圆
(x, y), radius = cv2.minEnclosingCircle(cnt)
(x, y, radius) = np.int0((x, y, radius))
# 或使用这句话取整：(x, y, radius) = map(int, (x, y, radius))
cv2.circle(img_color2, (x, y), radius, (0, 0, 255), 2)


# 6.拟合椭圆
ellipse = cv2.fitEllipse(cnt)
cv2.ellipse(img_color2, ellipse, (0, 255, 0), 2)

cv2.imshow('contours2', img_color2)
cv2.waitKey(0)


# 7.形状匹配
img = cv2.imread('shapes.jpg', 0)
_, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
image, contours, hierarchy = cv2.findContours(thresh, 3, 2)
img_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

cnt_a, cnt_b, cnt_c = contours[0], contours[1], contours[2]
print(cv2.matchShapes(cnt_b, cnt_b, 1, 0.0))  # 0.0
print(cv2.matchShapes(cnt_b, cnt_c, 1, 0.0))  # 2.17e-05
print(cv2.matchShapes(cnt_b, cnt_a, 1, 0.0))  # 0.418
'''
