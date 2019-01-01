#reading Video Stream

import cv2

cap = cv2.VideoCapture(0)
face_casc=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while True:
	ret,frame=cap.read()
	gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

	#faces will store different faces as list of tuples with face coordinates
	faces=face_casc.detectMultiScale(gray_frame,1.3,5)#1.3 is scaing factor

	if (ret== False):
		continue
	

	for (x,y,w,h) in faces:
		cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

	cv2	.imshow("VideoFrame",frame)
	#Wait for q to stop

	key_pressed=cv2.waitKey(1)&0xFF
	if key_pressed == ord('q'):
		break


cap.release()
cv2.destroyAllWindows()