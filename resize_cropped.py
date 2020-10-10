import numpy as np
import cv2
import os
## training and test sets split after cropping the images, and removing useless images that cover very less area of the face
train_files = [f for f in os.listdir(r'.\train_cropped')] # to resize the images from the training set
test_files = [f for f in os.listdir(r'.\test_cropped')] # to resize the images from the test set

for i in train_files:
    img = cv2.imread(r'.\\train_cropped\\'+i) # reading the image
    resized_img = cv2.resize(img, (320,320), interpolation=cv2.INTER_AREA) # resizing the image
    cv2.imwrite(".\\resize_train\\" +i, resized_img) # saving back the resized and cropped image in another directory
for i in test_files:
    img = cv2.imread(r'.\\test_cropped\\'+i) # reading the image
    resized_img = cv2.resize(img, (320,320), interpolation=cv2.INTER_AREA) # resizing the image
    cv2.imwrite(".\\resize_test\\" +i, resized_img) # saving back the resized and cropped image in another directory
