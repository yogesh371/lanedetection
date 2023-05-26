import cv2
import numpy as np

# Load the road image
image_path = "road.jpg"
image = cv2.imread(image_path)

# Resize the image to a desired width and height
desired_width = 800
desired_height = 600
resized_image = cv2.resize(image, (desired_width, desired_height))

# Convert the resized image to grayscale
gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to the grayscale image
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Perform Canny edge detection
edges = cv2.Canny(blur, 50, 150)

# Define a region of interest (ROI)
height, width = edges.shape[:2]
roi_vertices = [
    (0, height),
    (width / 2, height / 2),
    (width, height)
]
mask = np.zeros_like(edges)
cv2.fillPoly(mask, np.int32([roi_vertices]), 255)
masked_edges = cv2.bitwise_and(edges, mask)

# Perform Hough line transformation
lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=50)

# Draw the detected lines on the image
line_image = np.zeros_like(resized_image)
for line in lines:
    for x1, y1, x2, y2 in line:
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 5)

# Overlay the detected lines on the original image
result = cv2.addWeighted(resized_image, 0.8, line_image, 1, 0)

# Display the result
cv2.imshow("Lane Detection", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
