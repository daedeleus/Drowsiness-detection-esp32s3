from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2
import requests


URL = "http://192.168.4.1"
AWB = True

# Streaming function settings
def set_awb(url: str, awb: int=1):
    	try:
        	awb = not awb
        	requests.get(url + "/control?var=awb&val={}".format(1 if awb else 0))
    	except:
        	print("SET_QUALITY: something went wrong")
    	return awb

def set_quality(url: str, value: int=1, verbose: bool=False):
    	try:
        	if value >= 10 and value <=63:
            		requests.get(url + "/control?var=quality&val={}".format(value))
    	except:
        	print("SET_QUALITY: something went wrong")

def set_resolution(url: str, index: int=1, verbose: bool=False):
	try:
        	if verbose:
            		resolutions = "10: UXGA(1600x1200)\n9: SXGA(1280x1024)\n8: XGA(1024x768)\n7: SVGA(800x600)\n6: VGA(640x480)\n5: CIF(400x296)\n4: QVGA(320x240)\n3: HQVGA(240x176)\n0: QQVGA(160x120)"
            		print("available resolutions\n{}".format(resolutions))

        	if index in [10, 9, 8, 7, 6, 5, 4, 3, 0]:
            		requests.get(url + "/control?var=framesize&val={}".format(index))
        	else:
            		print("Wrong index")
	except:
        	print("SET_RESOLUTION: something went wrong")

# Function for eye parameters
def eye_aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear
	
# main code

set_resolution(URL, index=7)
thresh = 0.25
frame_check = 20
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")# Dat file is the crux of the code

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
cap=cv2.VideoCapture(URL + ":81/stream")
flag=0

while True:
	ret, frame=cap.read()
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	subjects = detect(gray, 0)
	for subject in subjects:
		shape = predict(gray, subject)
		shape = face_utils.shape_to_np(shape)#converting to NumPy Array
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
		ear = (leftEAR + rightEAR) / 2.0
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
		if ear < thresh:
			flag += 1
			print (flag)
			if flag >= frame_check:
				cv2.putText(frame, "****************ALERT!****************", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				cv2.putText(frame, "****************ALERT!****************", (10,325),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				#print ("Drowsy")
		else:
			flag = 0
	cv2.imshow("Frame", frame)
	
	key = cv2.waitKey(1)
	if key == ord('r'):
		idx = int(input("Select resolution index: "))
		set_resolution(URL, index=idx, verbose=True)

	elif key == ord('q'):
		break
	elif key == ord('a'):
		AWB = set_awb(URL, AWB)

cv2.destroyAllWindows()
cap.release() 
