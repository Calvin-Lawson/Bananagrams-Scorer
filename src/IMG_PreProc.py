import numpy as np
import matplotlib.pyplot as plt
import cv2

#image path
image_path = 'data/IMG_3009.JPG'
img = cv2.imread(image_path)
if img is None:
    raise ValueError("Error loading the image. Please check the file path.")


#grayscale
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#Gaussian Blur
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

#edge detection with canny
lower_threshold = 50
upper_threshold = 150
edges = cv2.Canny(blurred,lower_threshold,upper_threshold)

#Finding contours
countour_retrieval_mode = cv2.RETR_EXTERNAL 
#see https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#ga819779b9857cc2f8601e6526a3a5bc71
countour_approximation_method = cv2.CHAIN_APPROX_SIMPLE
#use simple for now, as it save memory
contours, _ = cv2.findContours(edges, countour_retrieval_mode, countour_approximation_method)
contour_img = img.copy()
cv2.drawContours(contour_img,contours,-1,(0,255,0),3)


#window resize
'''
cv2.namedWindow("GrayScale",cv2.WINDOW_NORMAL)
cv2.resizeWindow("GrayScale",600,600)

cv2.namedWindow("Blurred Image",cv2.WINDOW_NORMAL)
cv2.resizeWindow("Blurred Image",600,600)

cv2.namedWindow("Edges",cv2.WINDOW_NORMAL)
cv2.resizeWindow("Edges",600,600)
'''

cv2.namedWindow("Contours",cv2.WINDOW_NORMAL)
cv2.resizeWindow("Contours",600,600)

#show Images (un-comment to view a specific image type)
#cv2.imshow("GrayScale",gray)
#cv2.imshow("Blurred Image", blurred)
#cv2.imshow("Edges", edges)
cv2.imshow("Contours", contour_img)

#Awaits user input to close
cv2.waitKey(0)
cv2.destroyAllWindows()