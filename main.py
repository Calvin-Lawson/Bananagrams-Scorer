#imports
import easyocr
import matplotlib.pyplot as plt
import cv2

#image path
image_path = 'data/IMG_3010.JPG'
img = cv2.imread(image_path)
if img is None:
    raise ValueError("Error loading the image. Please check the file path.")

#tell easyocr to read english
reader = easyocr.Reader(['en'])
text = reader.readtext(img)

threshhold = 0.25

def draw_bounding_boxes(image, detections, threshold):

    for bbox, text, score in detections:

        if score > threshold:

            cv2.rectangle(image, tuple(map(int, bbox[0])), tuple(map(int, bbox[2])), (0, 255, 0), 5)

            #cv2.putText(image, text, tuple(map(int, bbox[0])), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.65, (255, 0, 0), 2)


draw_bounding_boxes(img,text,threshhold)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGBA))
plt.show()