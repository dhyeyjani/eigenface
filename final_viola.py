import numpy as np
import cv2
import os
files = [f for f in os.listdir(r'.\Yale_Dataset_Eigenface')] # Reading the files
count = 0 # count of properly detected images
image = [np.array(cv2.imread(r'.\\Yale_Dataset_Eigenface\\'+img)).flatten() for img in files] # array of flattened images
face_cascade = cv2.CascadeClassifier(r".\cascade.xml") #  from the cascade.xml trained from or dataset
for i in files:
    w_list = [] # width of images
    h_list = [] # height of images
    img = cv2.imread(r'.\\Yale_Dataset_Eigenface\\'+i)
    # print(img.shape)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # to convert it to grayscale
    faces = face_cascade.detectMultiScale(gray, 1.001, 4) # hese the detection happens
    for (x,y,w,h) in faces:
        w_list.append(w)
        h_list.append(h)
    pos = w_list.index(max(w_list)) # to find the maximum possible size of rectangle formed
    if w_list[pos]> 120 and h_list[pos] > 120: # if thi threshold is crossed, only then image is "detected"
        img = cv2.rectangle(img, (x,y),(x+w_list[pos],y+h_list[pos]),(255,0,0),2)
        cv2.imwrite(".\\cropped_dataset\\" +i, img[y:y+h_list[pos],x:x+w_list[pos]]) # cropping the image detected for further use in Eigenfaces
        count+=1
print("Accuracy:")
print(100*count/len(files)) # Accuracy
# cv2.imshow('img',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()