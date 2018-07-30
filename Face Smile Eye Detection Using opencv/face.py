# The objective of this program given is to detct object of interest
# in real time and to keep tracking of same .

import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

cap = cv2.VideoCapture(0)

while True:

	# read frames from camera
	ret, img = cap.read()

	# covert to gray scale of frame
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# Detect faces of different sizes in input image
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)

	
	for (x,y,w,h) in faces:
		# to draw rectangle
		cv2.rectangle(img, (x,y), (x+w,y+h),(255,255,0),2)
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = img[y:y+h, x:x+w]

		#Detect eyes of different sizes in input image
		eyes = eye_cascade.detectMultiScale(roi_gray)
		smile = smile_cascade.detectMultiScale(
			roi_gray,
			scaleFactor = 1.7,
			minNeighbors = 22,
			minSize = (25, 25),
			# flags = cv2.CASCADE_SMILE_IMAGE
			)

		# To draw rectangles in eyes
		for(ex,ey,ew,eh) in eyes:
			cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,127,255),2)

		
		#Set region of interest for smiles
		for(x, y, w, h) in smile:
			print "Found", len(smile), "smiles!!"
			cv2.rectangle(roi_color, (x, y), (x+w, y+h), (255, 0, 0), 1)


	cv2.imshow('img', img)

	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break


cap.release()

cv2.destroyAllWindows()
					