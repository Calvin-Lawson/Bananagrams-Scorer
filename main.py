#imports
import easyocr

#tell easyocr to read english
reader = easyocr.Reader(['en'], gpu = True)

#read the image
result = reader.readtext(r'C:\Users\Calvin\Documents\Coding Projects\Bananagrams-Scorer\data\IMG_3009.JPG', detail = 0, rotation_info=[90, 180, 270])

print(result)