import cv2
import numpy as np
# import matplotlib.pyplot as plt

def show_image(image, name) :
	cv2.namedWindow(name, cv2.WINDOW_NORMAL)
	cv2.imshow(name, image)
	cv2.resizeWindow(name, 600,600)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def point_finding(image) :
	points = []
	color_image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
	# for i in range(image.shape[0]) :
		# for j in range(image.shape[1]) :
			# if 
	for i, row in enumerate(image) : 
		# print row
		if 255 in row :
			j = int(np.where(row==255)[0][0])
			print("i,j",i,j)

			cv2.circle(color_image, (i,j), radius=10, color=(0,0,255), thickness=-1)
			break

	for i, row in enumerate(image) : 
		# print row
		if 255 in row :
			j = int(np.where(row==255)[0][0])
			print("i,j",i,j)

			cv2.circle(color_image, (i,j), radius=10, color=(0,0,255), thickness=-1)
			break

	show_image(color_image, "first white")


	return points



	
file = "image.jpeg"
img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
show_image(img, "normal")
kernel = np.ones((3,3), np.uint8)
# img_dilation3 = cv2.imread(file)
img_dilation = cv2.erode(img, kernel, iterations=1)
# img_dilation = cv2.dilate(img, kernel, iterations=1)
ret,img_dilation = cv2.threshold(img_dilation,127,255,cv2.THRESH_BINARY)
# show_image(img, "dilate")

img_dilation = cv2.bitwise_not(img_dilation)

show_image(img_dilation, "threshold")

# # show_image(img_dilation2, "dilation")
# (_,contours,_) = cv2.findContours(img_dilation2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# for contour in contours:
# 	rect = cv2.boundingRect(contour)
# 	cv2.rectangle(img_dilation3, (rect[0],rect[1]), (rect[2]+rect[0],rect[3]+rect[1]), (0,255,0), 2)

# show_image(img_dilation3, "rectangle")

# ============ Canny edge detector ===============

# edges = cv2.Canny(img_dilation, 100, 200)

# show_image(edges, "edges")

# l = point_finding(edges)
# print(l)

# ================================================

# dst = cv2.cornerHarris(img_dilation, 7, 5, 0.04)
# show_image(dst, "corners")

# ========= Blob detection ========================
detector = cv2.SimpleBlobDetector_create()
 
# Detect blobs.
keypoints = detector.detect(img_dilation)
# print len(keypoints)
# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(img_dilation, keypoints, np.array([]), (50,0,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

show_image(im_with_keypoints, "blob")
 
img = img_dilation.copy()
for x in range(1,len(keypoints)):
  img=cv2.circle(img, (np.int(keypoints[x].pt[0]),np.int(keypoints[x].pt[1])), radius=np.int(keypoints[x].size), color=(50), thickness=-1)

show_image(img, "blob")
# img = img_dilation.copy()
# # img = cv2.bitwise_not(img)


# (_,contours,_) = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# print "countours", len(contours)
# for contour in contours:
# 	rect = cv2.boundingRect(contour)
# 	cv2.rectangle(img, (rect[0],rect[1]), (rect[2]+rect[0],rect[3]+rect[1]), (50,50,50), 2)

# show_image(img, "rectangle")