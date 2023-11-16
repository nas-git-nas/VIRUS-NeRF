import cv2
import numpy as np

# # Read the infrared image
# image = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)

# # Apply Gaussian blur for noise reduction
# image = cv2.GaussianBlur(image, (5, 5), 0)

# # Apply thresholding
# threshold = 80
# _, image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

# # Perform morphological operations (erosion and dilation)
# kernel = np.ones((5, 5), np.uint8)
# image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

# # # Find contours and draw them
# # contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# # cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

# # Display the result
# cv2.imshow('Result', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



# Read the infrared image
depth = cv2.imread('img2.png', cv2.IMREAD_UNCHANGED)

# H, W = depth.shape

# print(f"pixel: {depth[H//2, W//2]}")


# Display the result
cv2.imshow('Result', depth)
cv2.waitKey(0)
cv2.destroyAllWindows()