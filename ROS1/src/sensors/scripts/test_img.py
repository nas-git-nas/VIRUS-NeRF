import numpy as np
import cv2 as cv 


img = np.array([[0,1,2],
                [3,4,5],
                [6,7,8]], dtype=np.uint16)

img = np.arange(27, dtype=np.uint16).reshape(3,3,3)

print(f"arr shape: {img.shape}, type: {img.dtype}")

# img = cv.cvtColor(arr, cv.CV_16U, cv.IMREAD_GRAYSCALE)
cv.imwrite("test_img.png", img)
img = cv.imread("test_img.png", cv.IMREAD_UNCHANGED)
# img = cv.cvtColor(img, cv.IMREAD_GRAYSCALE)

print(f"img shape: {img.shape}, type: {img.dtype}")
print(img)