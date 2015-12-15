import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

##############################################################################################################################################
#User adjustable parameters below

'''
General global variables AND 
Stage (1/3) : Equalized Image variables AND 
Stage (2/3) : Threshold Image variables
'''
#img_path = 'images/12.pgm'
img_dir = 'images'
#Stage (1/3) : Equalized Image variable
#adaptiveClipLimit = 30.0
#adaptiveTileGridSize = (40,40)
adaptiveClipLimit = 200
adaptiveTileGridSize = (100,100)
#Stage (2/3) : Threshold Image  variable
#thresholdLimit = 80    #high value= no blur, low value = full blur
#blur_aperture = 15
thresholdLimit = 55    #high value= no blur, low value = full blur
blur_aperture = 9

'''
Stage (3/3) : Final Image
Adjustable Hough Variables
    -minDist: Minimum distance between the center (x, y) coordinates of detected circles. 
            If the minDist is too small, multiple circles in the same neighborhood as the original 
            may be (falsely) detected. If the minDist is too large, then some circles may not be 
            detected at all.
    -param1: Gradient value used to handle edge detection in the Yuen et al. method.
    -param2: Accumulator threshold value for the cv2.HOUGH_GRADIENT method. The smaller 
            the threshold is, the more circles will be detected (including false circles). The 
            larger the threshold is, the more circles will potentially be returned.
    -minRadius: Minimum size of the radius (in pixels).
    -maxRadius: Maximum size of the radius (in pixels).

min_dist = 10
param1 = 40
param2 = 50
min_radius = 3
max_radius = 40
'''
min_dist = 10
param1 = 90
param2 = 40
min_radius = 3
max_radius = 40

##############################################################################################################################################
#DO NOT MODIFY ANYTHING BELOW THIS LINE
#I WILL FUCK YOU UP - TONY
##############################################################################################################################################
label_x = 'label for x'
label_y = 'label for y'
'''
Image sorting and calling functions onto the sorted images in an iterable fashion
    1. Finds images in a folder.
    2. Puts the images into an array.
    3. Each image is iterated through the different 'Stages'
'''
def main():
    #count_circles(img_path, True)
    file_list = []
    circle_count = []
    for file in os.listdir(img_dir):
        if file.endswith('.pgm'):
            file_list.append(file)
    file_list = sorted(file_list)
    for file in file_list:
        #print file                  #only to see if the array is print out correctly
        circle_count.append(count_circles(img_dir, file))
    
    f= open("test.csv", 'w')        #make the file name an argument
    f.write('#comment 1\n')
    f.write('#comment 1\n')
    f.write('%s, %s\n' %(label_x, label_y))
    for i in range (0, len(file_list)):
        file_split = file_list[i].split('.')
        f.write('%s,%d\n' %(file_split[0],circle_count[i]))
    f.close()



'''
Stage (1/3) : Equalized Image
    -Equlizes the brightness in the image. 
    -Even with a diffuser, the image doesn't have even brightness.
'''
def equalize_img_hist(img, adaptiveClipLimit, adaptiveTileGridSize):
    #Convert to BW
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #Apply adaptive histogram equalization
    img_lab = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=adaptiveClipLimit, tileGridSize=adaptiveTileGridSize)
    #change if you want to use colored images
    img_adaptHist = clahe.apply(img_gray)

    return img_adaptHist

'''
Stage (2/3) : Threshold Image
    -Doesn't do much; just setting a parameter using a cv2 module function called medianBlur
'''
def blur_img(img, aperture):
    img_blur = cv2.medianBlur(img,aperture)

    return img_blur

'''
Stage (3/3) : Final Image
This is where shit goes down
'''
def count_circles(img_dir,file, display_image = False, save_image = True):
    print 'Processing ',file
    # Read image
    img_path = os.path.join(img_dir, file)
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
    else:
        circles = []

        #print "Circles found: ", len(circles)



    #Displays a matplotlib image of 4 images: original image, equalized image, threshold image and final image
    if display_image:
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

    #Saves a matplotlib image of 4 images: original image, equalized image, threshold image and final image
    if save_image:
        # Result Display
        plt.subplot(221),plt.imshow(img,cmap = 'gray')
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(222),plt.imshow(img_adaptHist,cmap = 'gray')
        plt.title('Stage (1/3) : Equalized Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(223),plt.imshow(im_th2,cmap = 'gray')
        plt.title('Stage (2/3) : Threshold Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(224),plt.imshow(output,cmap = 'gray')
        plt.title('Stage (3/3) : Final Image'), plt.xticks([]), plt.yticks([])
        fig1 = plt.gcf()
        #plt.show()
        plt.draw()

        #dirpath to a folder of images generated by matplotlib
        matplotlib_generated_img_path = 'generated-images/'
        fig1.savefig('%s%s.png' %(matplotlib_generated_img_path ,file) , dpi=1000)


    return len(circles)
#run the script
if __name__ == "__main__":
    main()
