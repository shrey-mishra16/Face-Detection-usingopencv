#reading Video Stream

import cv2

cap = cv2.VideoCapture(0)

while True:
	ret,frame=cap.read()
	#frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

	if (ret== False):
		continue
	cv2	.imshow("VideoFrame",frame)

	#Wait for q to stop

	key_pressed=cv2.waitKey(1)&0xFF
	if key_pressed == ord('q'):
		break


cap.release()
cv2.destroyAllWindows()