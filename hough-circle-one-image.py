import cv2
import numpy as np
from matplotlib import pyplot as plt

def equalize_img_hist(img, adaptiveClipLimit, adaptiveTileGridSize):
    #Convert to BW
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #Apply adaptive histogram equalization
    img_lab = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=adaptiveClipLimit, tileGridSize=adaptiveTileGridSize)
    img_adaptHist = clahe.apply(img_gray)           #change if you want to use colored images

    return img_adaptHist

def blur_img(img, aperture):
    img_blur = cv2.medianBlur(img,aperture)

    return img_blur

img_path = 'images/012.pgm'
adaptiveClipLimit = 30.0
adaptiveTileGridSize = (40,40)
thresholdLimit = 80
blur_aperture = 15

# Hough Variables
min_dist = 10
param1 = 40
param2 = 50
min_radius = 3
max_radius = 100

# Read image
img = cv2.imread(img_path)
output = img.copy()

# Filter image to get better edge detection results
img_adaptHist = equalize_img_hist(img, adaptiveClipLimit, adaptiveTileGridSize)
th2, im_th2 = cv2.threshold(img_adaptHist, thresholdLimit, 255, cv2.THRESH_BINARY_INV)
img_blur = blur_img(im_th2, blur_aperture)

# Find circles via Hough transform
circles = cv2.HoughCircles(img_blur, cv2.cv.CV_HOUGH_GRADIENT, 2, min_dist, np.array([]), param1, param2, min_radius, max_radius)

# Add circles to output image
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")

    for (x, y, r) in circles:
        cv2.circle(output, (x,y), r, (0, 255, 0), 4)
        cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
    print "Circles found: ", len(circles)
# Result Display
plt.subplot(221),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(222),plt.imshow(img_adaptHist,cmap = 'gray')
plt.title('Stage (1/3) : Equalized Image'), plt.xticks([]), plt.yticks([])
plt.subplot(223),plt.imshow(im_th2,cmap = 'gray')
plt.title('Stage (2/3) : Threshold Image'), plt.xticks([]), plt.yticks([])
plt.subplot(224),plt.imshow(output,cmap = 'gray')
plt.title('Stage (3/3) : Final Image'), plt.xticks([]), plt.yticks([])

plt.show()

f = open('total_bead_count.csv', 'a')
f.write('%s\n' %(len(circles)))
f.close()