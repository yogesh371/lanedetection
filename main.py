import cv2

# Load the pre-trained YOLOv4 model
model_weights = "yolov4.weights"
model_config = "yolov4.cfg"
net = cv2.dnn.readNetFromDarknet(model_config, model_weights)

# Load the class labels
class_labels = []
with open("coco.names", "r") as f:
    class_labels = [line.strip() for line in f.readlines()]

# Open the default webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the frame from the webcam
    ret, frame = cap.read()

    # Create a blob from the input frame
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)

    # Set the input blob to the network
    net.setInput(blob)

    # Forward pass through the network
    detections = net.forward()

    # Process the detections
    for detection in detections:
        scores = detection[5:]
        class_id = scores.argmax()
        confidence = scores[class_id]

        if confidence > 0.5:
            # Get the bounding box coordinates
            box = detection[0:4] * [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]]
            (x, y, w, h) = box.astype("int")

            # Draw the bounding box and label on the frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"{class_labels[class_id]}: {confidence:.2f}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with object detections
    cv2.imshow("Object Detection", frame)

    # Check for the 'q' key press to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()
