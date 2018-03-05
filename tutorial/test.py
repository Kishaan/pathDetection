import cv2
import numpy as np
import matplotlib.pyplot as plt

# ========= read image ============================
img = cv2.imread("goku.jpg", cv2.IMREAD_COLOR)
# cv2.IMREAD_GRAYSCALE or 0 for grascale
# cv2.IMREAD_COLOR or 1 for rgb image
# -1 for rgba image

# ========== draw stuff on image ====================
# color in cv2 is in form of bgr and not rgb
# param are (image, starting point, ending point, color, linewidth)
cv2.line(img, (0,0), (150, 150), (255, 255, 255), 15)

# param are (image, top left, bottom right, color, linewidth)
cv2.rectangle(img, (15, 25), (200, 150), (0,0,255), 5)

# -1 as linewidth fill the shape
# param are (image, center, radius, color, linewidth)
cv2.circle(img, (100,63), 50, (0,255,0), -1)

# polygon
pts = np.array([[10,5], [20,30], [70, 20], [50,10]], np.int32)
# pts = pts.reshape((-1, 1, 2))
# third argument is whether the first and last point should be connected or not
cv2.polylines(img, [pts], True, (0,255,255), 3)

# ========= write text on image =====================
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, "hello", (0,130), font, 1, (200, 255, 255), 2, cv2.LINE_AA)

# ============== cv2 to show the image ===============
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ============= cv2 to save the image ================
# cv2.imwrite("goku_gray.jpg", img)

# ======= matplotlib to show the image ===============
# plt.imshow(img, cmap="gray", interpolation="bicubic")
# plt.plot([50,100], [80, 100], "c", linewidth=5)
# plt.show()


# ============ capture video and show it ======================
# cap = cv2.VideoCapture(0)

# ==== save video things ================
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter("output.avi", fourcc, 20.0, (640, 480))


# while True :
# 	ret, frame = cap.read()
# 	# out.write(frame)
# 	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# 	cv2.imshow("frame", frame)
# 	cv2.imshow("gray", gray)

# 	if cv2.waitKey(1) & 0xFF == ord('q') :
# 		break

# cap.release()
# # out.release()
# cv2.destroyAllWindows()