import cv2
import numpy as np

# read an image
img = cv2.imread('images/5.jpg')
# resize image
img = cv2.resize(img,(img.shape[1] // 2,img.shape[0] // 2))
# change BGR format to GRAY
grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# use model for recognising faces
faces = cv2.CascadeClassifier('trained_model/faces.xml')
result = faces.detectMultiScale(grey,1.05,7)
# highlight faces
for (x,y,w,h) in result:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

#img = cv2.Canny(img, 100, 200)
#kernel = np.ones((5,5),np.uint8)
#img = cv2.dilate(img, kernel, iterations=1)
#img = cv2.erode(img, kernel, iterations=1)

cv2.imshow('img', img)
cv2.waitKey(0)