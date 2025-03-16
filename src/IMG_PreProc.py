import numpy as np
import matplotlib.pyplot as plt
import cv2


def cv_plot(name:str = None, img = None, scale = (600,600)):
    if name is None or img is None:
            raise OSError("Set all arguments")
    cv2.namedWindow(name,cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name,scale[0],scale[1])
    cv2.imshow(name,img)

#image path
_image_path = 'data/IMG_3009.JPG'
_img = cv2.imread(_image_path)
if _img is None:
    raise ValueError("Error loading the image. Please check the file path.")


#grayscale
gray = ('Grayscale',cv2.cvtColor(_img,cv2.COLOR_BGR2GRAY))

#Gaussian Blur
blurred = ('Blurred',cv2.GaussianBlur(gray[1], (5, 5), 0))

#edge detection with canny
_lower_threshold = 90
_upper_threshold = 125
canny_edges = ('Canny Edges',cv2.Canny(blurred[1],_lower_threshold,_upper_threshold))
adadptive_threshold = ('AdvThold',cv2.adaptiveThreshold(blurred[1], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,2))

#denoise 
kernal = np.ones((3,3),np.uint8)
denoised = ('Denoised',cv2.morphologyEx(adadptive_threshold[1],cv2.MORPH_OPEN,kernal))
 

#Finding contours
_countour_retrieval_mode = cv2.RETR_EXTERNAL 
_countour_approximation_method = cv2.CHAIN_APPROX_SIMPLE
#see https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#ga819779b9857cc2f8601e6526a3a5bc71

#use simple for now, as it save memory
_contours, _ = cv2.findContours(denoised[1], _countour_retrieval_mode, _countour_approximation_method)
_filtered_contours = [cnt for cnt in _contours if cv2.contourArea(cnt) > 50]
_contour_img = _img.copy()
cv2.drawContours(_contour_img, _filtered_contours, -1, (0,255,0), 2)
contour = ('Filterd Contours',_contour_img)

#show Images (un-comment loop to view all or select a specfic image type)
Images = [gray,blurred,adadptive_threshold,denoised,contour]

'''for i in Images:
     cv_plot(i[0],i[1])  '''

cv_plot(Images[-1][0],Images[-1][1])

#Awaits user input to close (press any keyboard key not the "x" button to close)
cv2.waitKey(0)
cv2.destroyAllWindows()