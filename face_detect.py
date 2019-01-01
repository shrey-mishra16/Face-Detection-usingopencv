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

def knn(train,query_x,k=5):
    dist = []

    for i in range (train.shape[0]):
    	#Get The Vector And Label
    	ix=train[i,:-1]
    	iy=train[i,-1]
    	#Compute Distance From test Point
    	d=dist(test,ix)
    	dist.append([d,iy])
    #Sort Based On Distance And Get Top k
    dk=sorted(dist,key=lambda x:x[0])[:k]
    #Retrieve Only Te Labels
    labels = np.array(dk)[:,-1]

    #Get Frequencies Of Each Label
    op=np.unique(labels,return_counts=True)

    #Find max Frequency And Corresponding Labels
    index = np.argmax(op[1])
    return op[1][index]	


# Init Camera

cap= cv2.VideoCaptur(0)

#Face Detection

face_cascade = cv2.CascadeClassifier("haar_cascade_frontalface_default.xml")

skip = 0
dataset_path = './face_dataset/'
face_data = []

label=[]

class_id = 0#Labels For the Given File
names = {}#Mappintg btw id-file

#Data Preparation

for fx in os.listdir(dataset_path): # For Giving Path
	if fx.endswith('.npy'):

		data_item = np.load(dataset_path+fx)
		face_data.append(data_item)

		#Create Labels For Class

		target = class_id*np.ones((data_item.shape[0],))
		class_id+=1
		labels.append(target)

face_dataset = np.concatenate(face_data,axis=0)
face_labels= np.concatenate(labels,axis=0).reshape((-1,1))

print(face_dataset.shape)
print(face_labels.shape)

trainset = np.concatenate((face_dataset,face_labels),axis=1)
print (trainset.shape)

#Testing

font = cv2.FONT_HERSHEY_SIMPLEX

while True :
	ret,face=cap.read()
	gray_frame=cv2.cvtColor(frame,cv2.cvtColor_BGR2GRAY)

	#scaling,detecting multi faces
	faces = face_cascade.detectMultiScale(gray_frame,1.3,5)

	if (ret==False):
		continue
	skip+=1	