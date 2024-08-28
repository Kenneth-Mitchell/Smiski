import cv2
import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# Load the ONNX model
onnx_model_path = 'yolov5/runs/train/exp2/weights/best.onnx'
session = ort.InferenceSession(onnx_model_path)

# Define input size for the model
input_size = (640, 640)

# Function to preprocess the image for YOLO
def preprocess(image):
    image_resized = cv2.resize(image, input_size)
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB) / 255.0
    image_tensor = np.transpose(image_rgb, (2, 0, 1)).astype(np.float32)
    return np.expand_dims(image_tensor, axis=0)

import numpy as np

def postprocess(output, conf_threshold=0.5):
    detections = output[0].squeeze(0)  # Shape: (25200, 6)
    
    # If detections are empty, return empty lists
    if detections.size == 0:
        return np.array([]), np.array([]), np.array([])
    
    boxes = detections[:, :4]
    objectness = detections[:, 4]
    class_scores = detections[:, 5:]
    
    # Compute the confidence score for each detection
    scores = objectness * np.max(class_scores, axis=1)
    classes = np.argmax(class_scores, axis=1)
    
    # Apply confidence threshold
    indices = np.where(scores > conf_threshold)[0]
    
    # Return filtered results
    return boxes[indices], scores[indices], classes[indices]



def scale_boxes(boxes, frame_shape, input_size):
    h, w = frame_shape[:2]  # Original frame dimensions
    input_h, input_w = input_size  # Model input dimensions
    
    # YOLOv5 outputs boxes in the format [x_center, y_center, width, height]
    # Scale x_center and width
    boxes[:, 0] = boxes[:, 0] * w / input_w  # x_center
    boxes[:, 2] = boxes[:, 2] * w / input_w  # width
    
    # Scale y_center and height
    boxes[:, 1] = boxes[:, 1] * h / input_h  # y_center
    boxes[:, 3] = boxes[:, 3] * h / input_h  # height
    
    # Convert from [x_center, y_center, width, height] to [x1, y1, x2, y2]
    boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
    boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
    boxes[:, 2] = boxes[:, 0] + boxes[:, 2]  # x2
    boxes[:, 3] = boxes[:, 1] + boxes[:, 3]  # y2
    
    # Ensure boxes are within the frame dimensions
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h)
    
    return boxes


def find_iphone_camera():
    index = 0
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.read()[0]:
            break
        
        # Get the camera name
        camera_name = cap.getBackendName()
        
        # Check if it's the iPhone camera (adjust the name as needed)
        if "iPhone" in camera_name:
            cap.release()
            return index
        
        cap.release()
        index += 1
    
    return None

# In your main code:
iphone_index = find_iphone_camera()
if iphone_index is not None:
    cap = cv2.VideoCapture(iphone_index)
else:
    print("iPhone camera not found. Using default camera.")
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Camera not accessible.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to capture video.")
        break

    # Preprocess the frame
    input_tensor = preprocess(frame)

    # Run inference
    outputs = session.run(None, {'images': input_tensor})
    
    # Post-process the outputs
    boxes, scores, classes = postprocess(outputs)
    
    # Scale boxes to pixel coordinates
    boxes = scale_boxes(boxes, frame.shape, input_size)

    # Draw bounding boxes on the frame for valid detections
    for box, score, cls in zip(boxes, scores, classes):
        x1, y1, x2, y2 = box.astype(int)
        
        # Ensure coordinates are within frame dimensions
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        
        if x1 < x2 and y1 < y2:
            print(f'Class: {int(cls)} Conf: {score:.2f}')
            print(f'Box: {x1, y1, x2, y2}')
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'Class: {int(cls)} Conf: {score:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('YOLOv5 Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
