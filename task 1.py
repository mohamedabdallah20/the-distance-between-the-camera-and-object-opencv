from imutils import paths
import numpy as np
import imutils
from imutils import perspective
from scipy.spatial.distance import euclidean
import cv2

# --------------to scal the image----------------------------------------
def imgscal(img, scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return img
# ---------------mid point betwin two points-----------------------------
def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

# ----------------max between two numbers--------------------------------
def findmax(_1 , _2):
	if _1 > _2 :
		return _1
	else :
		return _2
# ---the formula to calculate the distance between the object and the camera-----
def distance_to_camera(knownWidth, focalLength, perWidth):
	return (knownWidth * focalLength) / perWidth
# ------------------track bars---------------------------
def empty(a):
    pass
cv2.namedWindow("trackBars")
cv2.resizeWindow("trackBars",500,200)
cv2.createTrackbar('Hue_Min', "trackBars" , 0,255,empty)
cv2.createTrackbar('Hue_max', "trackBars" , 151,151,empty)
cv2.createTrackbar('S_Min', "trackBars" , 18,205,empty)
cv2.createTrackbar('S_Max', "trackBars" , 135,135,empty)
cv2.createTrackbar('V_Min', "trackBars" , 142,255,empty)
cv2.createTrackbar('V_max', "trackBars" , 255,255,empty)

# --------find tallest edge/pixel to fit any rotate to an object-----
def findWidthPerPixel(box,img) :
	box = np.array(box)
	box = perspective.order_points(box.astype(int))
	(tl, tr, br, bl) = box
	wm1 = midpoint(tl, tr)
	wm2 = midpoint(bl, br)
	cv2.line(img, (int(wm1[0]),int(wm1[1])), (int(wm2[0]),int(wm2[1])), (0, 0, 255))
	w = euclidean(wm1, wm2)
	hm1 = midpoint(tl, bl)
	hm2 = midpoint(tr, br)
	cv2.line(img, (int(hm1[0]), int(hm1[1])), (int(hm2[0]), int(hm2[1])), (0, 0, 255))
	h = euclidean(hm1, hm2)
	return findmax(h,w)
# -----------------------------------------------------------------
KNOWN_DISTANCE = 22.0
KNOWN_WIDTH = 4
FOCAL_LENGTH = 676.5
WIDTH_PERPIXEL = None
# -----------------------------------------------------------------
hue_min = 0
hue_max = 0
s_min = 0
s_max = 0
v_min = 0
v_max = 0
# -------------------------------------------------------------------
while True:
	img = cv2.imread('img3.jpg')
	img = imgscal(img,20)
	imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
	hue_min= cv2.getTrackbarPos('Hue_Min', "trackBars")
	hue_max = cv2.getTrackbarPos('Hue_max', "trackBars")
	s_min = cv2.getTrackbarPos('S_Min', "trackBars")
	s_max = cv2.getTrackbarPos('S_Max', "trackBars")
	v_min = cv2.getTrackbarPos('V_Min', "trackBars")
	v_max = cv2.getTrackbarPos('V_max', "trackBars")

	lower = np.array([hue_min, s_min, v_min])
	upper = np.array([hue_max, s_max, v_max])
	Mask = cv2.inRange(img, lower, upper)


	cnts , _ = cv2.findContours(Mask , cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

	result = cv2.bitwise_and(img, img, mask=Mask)

	if len(cnts) > 0 :
		c = max(cnts, key=cv2.contourArea)
		rect = cv2.minAreaRect(c)
		box = cv2.cv.BoxPoints(rect) if imutils.is_cv2() else cv2.boxPoints(rect)
		box = np.int0(box)
		cv2.drawContours(img, [box], -1, (0, 255, 0), 2)

		WIDTH_PERPIXEL = findWidthPerPixel(box,img)
		FOCAL_LENGTH = ((WIDTH_PERPIXEL) * KNOWN_DISTANCE) / KNOWN_WIDTH
		print(FOCAL_LENGTH)
		dis = distance_to_camera(KNOWN_WIDTH,FOCAL_LENGTH,WIDTH_PERPIXEL)
		print(dis)

	cv2.putText(img, "chose the focal length and presss ESC ", (200, 50),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
	cv2.putText(img, "{:.1f}".format(FOCAL_LENGTH), (230, 70),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

	# cv2.imshow("frame" , result)
	cv2.imshow("original" , img)
	key = cv2.waitKey(1)
	if key == 27 :
		break

cv2.destroyAllWindows()

# -----------------------------------------------------------------

cap = cv2.VideoCapture('http://192.168.1.6:8080/video')
# cap = cv2.VideoCapture(0)
# ------------------------------------------------------------------
cv2.namedWindow("trackBars2")
cv2.resizeWindow("trackBars2",500,200)
cv2.createTrackbar('Hue_Min', "trackBars2" , hue_min,255,empty)
cv2.createTrackbar('Hue_max', "trackBars2" , hue_max,hue_max,empty)
cv2.createTrackbar('S_Min', "trackBars2" , s_min,205,empty)
cv2.createTrackbar('S_Max', "trackBars2" , s_max,s_max,empty)
cv2.createTrackbar('V_Min', "trackBars2" , v_min,255,empty)
cv2.createTrackbar('V_max', "trackBars2" , v_max,v_max,empty)
# ------------------------------------------------------------------
while True :
	# _ , img = cap.read()
	# img = imgscal(img,40)
	img = cv2.imread('img.jpg')
	img = imgscal(img, 20)
	imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
	hue_min= cv2.getTrackbarPos('Hue_Min', "trackBars2")
	hue_max = cv2.getTrackbarPos('Hue_max', "trackBars2")
	s_min = cv2.getTrackbarPos('S_Min', "trackBars2")
	s_max = cv2.getTrackbarPos('S_Max', "trackBars2")
	v_min = cv2.getTrackbarPos('V_Min', "trackBars2")
	v_max = cv2.getTrackbarPos('V_max', "trackBars2")

	lower = np.array([hue_min, s_min, v_min])
	upper = np.array([hue_max, s_max, v_max])
	Mask = cv2.inRange(img, lower, upper)
	cnts , _ = cv2.findContours(Mask , cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	result = cv2.bitwise_and(img, img, mask=Mask)
	if len(cnts) > 0 :
		c = max(cnts, key=cv2.contourArea)

		rect = cv2.minAreaRect(c)
		box = cv2.cv.BoxPoints(rect) if imutils.is_cv2() else cv2.boxPoints(rect)
		box = np.int0(box)
		cv2.drawContours(img, [box], -1, (0, 255, 0), 2)

		WIDTH_PERPIXEL = findWidthPerPixel(box, img)

		dis = distance_to_camera(KNOWN_WIDTH,FOCAL_LENGTH,WIDTH_PERPIXEL)
		cv2.putText(img, "{:.1f}cm".format(dis), (result.shape[1] - 200, result.shape[0] - 20),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

		print(dis)
	# cv2.imshow("frame" , result)
	cv2.imshow("original" , img)
	key = cv2.waitKey(1)
	if key == 27 :
		break

cap.release()
cv2.destroyAllWindows()
