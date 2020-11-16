""""
Created by Synapse.
This is an open source project that you can use in your own projects not in comercial use!
This script is not using any cascade like the one for Russian plates only! its for any license plate but be careful on resolution and scan area!


-> you can perform any action like adding a live video source (ex: webcam, videcapture software using the opencv import functions);
-> you can verify the license plate numbers for whatever you need to do like opening an relee or logging informations for your house or park lot;
""""



# import all of your other libraries like adafruit or whatever!
import cv2
import os
import pytesseract
import matplotlib.pyplot as plt

# insert tesseract path for your configuration
pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

images_dir = "Resouces"
image_files = os.listdir(images_dir)

image_path = "{}/{}".format(images_dir, "car_1.jpg")

image = cv2.imread(image_path)
imgCompare = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def plot_images(img1, img2, title1="", title2=""):
    fig = plt.figure(figsize=[15,15])
    ax1 = fig.add_subplot(121)
    ax1.imshow(img1, cmap="gray")
    ax1.set(xticks=[], yticks=[], title=title1)

    ax2 = fig.add_subplot(122)
    ax2.imshow(img2, cmap="gray")
    ax2.set(xticks=[], yticks=[], title=title2)

plot_images(image, gray)

blur = cv2.bilateralFilter(gray, 11,90, 90)

edges = cv2.Canny(blur, 30, 200)

plot_images(blur, edges)

cnts, new = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
image_copy = image.copy()
_ = cv2.drawContours(image_copy, cnts, -1, (255,0,255),2)
plot_images(image, image_copy)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]

image_copy = image.copy()
_ = cv2.drawContours(image_copy, cnts, -1, (255,0,255),2)

plot_images(image, image_copy)

plate = None
for c in cnts:
    # declare your area of working (center of your images/video/webcam)
    perimeter = cv2.arcLength(c, True)
    edges_count = cv2.approxPolyDP(c, 0.02 * perimeter, True)
    if len(edges_count) == 4:
        x,y,w,h = cv2.boundingRect(c)
        plate = image[y:y+h, x:x+w]
        break

# saving the latest license plate for logging! Adapt for your own situation!
cv2.imwrite("plate.png", plate)

plot_images(plate, plate)

# ocr detection
text = pytesseract.image_to_string(plate, lang="eng")

import re
cleanText = re.sub('\W+?','', text)

print(cleanText)

w, h, _ = imgCompare.shape

# tesing the outputs with images
cv2.putText(imgCompare, cleanText, (int(w/4), int(h/3)), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)
cv2.imshow('Plate', plate)
cv2.imshow('Car', imgCompare)
cv2.waitKey(0)