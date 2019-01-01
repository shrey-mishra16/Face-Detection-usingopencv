"""
Tasks :
1. Read Video Stream Using OpenCV
2.Extract Faces Out
3. Load Training Data As Numpy Files
4.Use KNN To Find Prediction
5.Map The Predicted Id To Nme Of Person
6. Display Predictions

"""

import cv2
import numpy as np
import os

# The K NN Funtion To Predict Int id

def dist(x1,x2): # Function For Euclidian Distance
	return np.sqrt(np.sum((x1-x2)**2))

def knn(X,Y,query_x,k=5):
    m = X.shape[0]
    
    d_list = []
    
    for i in range(m):
        
        d = dist(X[i],query_x)
        d_list.append((d,Y[i]))
        
    d_list = sorted(d_list,reverse=False)
    d_list = np.array(d_list[:k])
    d_list = d_list[:,1]
    freq = np.unique(d_list,return_counts=True) 
    idx = np.argmax(freq[1])
    pred = freq[0][idx]
    
    return int(pred)


# Init Camera

cap= cv2.VideoCaptur(0)

#Face Detection

face_cascade = cv2.CascadeClassifier("haar_cascade_frontalface_default.xml")

skip = 0
dataset_path = './face_dataset/'
face_data = []

label=[]

class_id = 0
names = {}

#Data Preparation

for fx in os.listdir(dataset_path): # For Giving Path
	if fx.endswith('.npy'):

		data_item = np.load(dataset_path+fx)
		face_data.append(data)

		#Create Labels For Class

		target = class_id*np.ones((data_item.shape[0],))
		class_id+=1
		labels.append(target)

face_dataset = np.concatenate(face_data,axis=0)
face_labels= np.concatenate(labels,axis=0)


"""
while True :
	ret,face=cap.read()
	gray_frame=cv2.cvtColor(frame,cv2.cvtColor_BGR2GRAY)

	#scaling
	faces = face_cascade.detectMultiScale(gray_frame,1.3,5)

	if (ret==False):
		continue
	skip+=1	