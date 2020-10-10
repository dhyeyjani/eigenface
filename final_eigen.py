import numpy as np
import cv2
import os
from sklearn.metrics import confusion_matrix as CM
import matplotlib.pyplot as plt
## Image files from the dataset
files = [f for f in os.listdir(r'.\resize_train')]
## Extracting the labels from the file names
labels = sorted(list(set([f.split(".")[0] for f in os.listdir(r'.\resize_train')])))
## Creating an array of flattened images from the dataset

X_train = np.array([np.array(cv2.imread(r'.\\resize_train\\'+img,0)).flatten() for img in files])
avg_mat = np.mean(X_train, axis=0) # average of all the training images
# plt.imshow(avg_mat.reshape(320,320), cmap = 'gray')
# plt.show()
A = np.array([i - avg_mat for i in X_train]) # subtracting average from all training images

## plotting the "mean subtracted image" for visualizing
# plt.imshow(A[0].reshape(320,320), cmap = 'gray')
# plt.show()

cov_A_red = np.dot(A, A.T) # taking the covariance matrix having a size of no. of images x no. of images
# print(cov_A_red.shape)
eig_val, eig_vec = np.linalg.eig(cov_A_red) # taking the eigenvectors and eigenvalues from the reduced covariance matrix
## Sorting the eigenvectors by the eigenvalues in descending order
key = eig_val.argsort()[::-1] 
val_sort = eig_val[key]
vec_sort = eig_vec[:,key]
vec_mul = np.dot(vec_sort.T, A) # getting the eigenvectors for the original caovariance matrix
# plt.imshow(np.absolute(vec_mul[0]).reshape(320,320), cmap = 'gray')
# plt.show()
norm_vec_mul = np.array([x/np.linalg.norm(x, ord=2, axis=0, keepdims=True) for x in vec_mul]) # normalizing the eigenvectors

feat = 27 # no. of features taken
weights = np.array([[np.dot(norm_vec_mul[j].T,A[i]) for j in range(feat)] for i in range(X_train.shape[0])]) # weights extracted from the training set

test_files = [f for f in os.listdir(r'.\resize_test')] # reading the test images
test_labels = [f.split(".")[0] for f in os.listdir(r'.\resize_test')] # extracting the labels from the test images
X_test = np.array([np.array(cv2.imread(r'.\\resize_test\\'+img,0)).flatten() for img in test_files]) # array of flattened test images
phi = np.array([img - avg_mat for img in X_test]) # difference with average image vector of trained images
img_weight = np.array([np.array([np.dot(test_phi.T,norm_vec_mul[i]) for i in range(feat)]) for test_phi in phi]) # calculating th weights of test images
dist = np.array([[np.linalg.norm(test_weight-weight) for weight in weights] for test_weight in img_weight]) # calculating the distance with the trained and test weights

## To calculate the threshold of 14000
# xx = [np.amin(x) for x in dist]
# print(xx)
# print(max(xx))

detected_labels = [] # detected images
anomalies = [] # detected anomalies
cnt = 0 

# flagging an image if it is an anomaly and moving it in anomaly set
for i in dist: 
    if np.amin(i)< 14000:   
        detected_labels.append(labels[int(np.argmin(i)/8)])
        cnt+=1
    else:
        anomalies.append(test_labels[cnt])
        cnt+=1
for i in anomalies:
    if i in test_labels:
        test_labels.remove(i)
accuracy = 0

# matching the matched lable with the actual label
for i in range(len(detected_labels)):
    print("Image of " + test_labels[i] + " detected as " + detected_labels[i])
    if test_labels[i]==detected_labels[i]:
        accuracy+=1
print("The accuracy is: " + str(100*accuracy/len(test_labels))) # printing the accuracy
print("Images detected outside the dataset:")
print(anomalies)
print("The confusion matrix is:") # printing the confusion matrix
print(CM(test_labels, detected_labels, labels=test_labels))

# visualizing the confusion matrix
plt.imshow(CM(test_labels, detected_labels, labels=test_labels), cmap='viridis')
plt.colorbar()
plt.show()