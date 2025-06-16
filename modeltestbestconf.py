import cv2
import numpy as np
from ultralytics import YOLO

# Load trained YOLOv8 model
model = YOLO("runs\\detect\\train4\\weights\\best.pt")

class_names = ["Seal", "Signature"]  

# Load image
image_path = "output_images\page_1.jpg"  # Replace with your test image
image = cv2.imread(image_path)

# Perform inference
results = model(image)

# Dictionary to store the best detection for each class
best_detections = {}

# Extract detections
for result in results:
    if result.boxes is not None:
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes (x1, y1, x2, y2)
        scores = result.boxes.conf.cpu().numpy()  # Confidence scores
        classes = result.boxes.cls.cpu().numpy().astype(int)  # Class IDs

        for i in range(len(scores)):
            class_id = classes[i]
            if class_id not in best_detections or scores[i] > best_detections[class_id]["score"]:
                best_detections[class_id] = {
                    "box": boxes[i],
                    "score": scores[i]
                }

# Draw bounding boxes on image
for class_id, detection in best_detections.items():
    x1, y1, x2, y2 = map(int, detection["box"])
    score = detection["score"]
    class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"

    # Draw rectangle
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Put class name and confidence score
    label = f"{class_name}: {score:.2f}"
    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Save the output image
output_path = "output.jpg"
cv2.imwrite(output_path, image)
print(f"Output saved as {output_path}")
