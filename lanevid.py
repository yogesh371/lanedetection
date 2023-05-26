import cv2
import numpy as np

# Path to the video file
video_path = "lane.mp4"
#video_path = "driving.mp4"
#video_path = "driving1.mp4"
# Create a VideoCapture object to read the video file
cap = cv2.VideoCapture(video_path)

# Define the threshold value for curve detection
threshold = 0.5  # Adjust the value according to your needs

while True:
    # Read the current frame from the video
    ret, frame = cap.read()

    # Check if the frame was successfully read
    if not ret:
        break

    # Resize the frame to a desired width and height
    desired_width = 800
    desired_height = 600
    resized_frame = cv2.resize(frame, (desired_width, desired_height))

    # Convert the resized frame to grayscale
    gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

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

    # Check if lines are detected
    if lines is not None:
        # Draw the detected lines on the frame
        line_image = np.zeros_like(resized_frame)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 5)

            # Calculate slope
            slope = (y2 - y1) / (x2 - x1) if x2 != x1 else 0

            # Determine direction based on slope
            if abs(slope) > threshold:
                if slope < 0:
                    direction = "Turn left"
                else:
                    direction = "Turn right"
            else:
                direction = "Continue straight"

            # Print suggested direction
            print("Direction: {}".format(direction))

        # Overlay the detected lines on the original frame
        result = cv2.addWeighted(resized_frame, 0.8, line_image, 1, 0)
    else:
        # If no lines are detected, display the original frame
        result = resized_frame

    # Display the result
    cv2.imshow("Lane Detection", result)

    # Check for the 'q' key to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object and close all windows
cap.release()
cv2.destroyAllWindows()
