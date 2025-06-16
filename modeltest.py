from ultralytics import YOLO
import cv2

# Load the trained YOLOv8 model
model = YOLO("/home/labadmin/Desktop/CDIT/YOLO/d:/Projects/cdit/YOLO/runs/seal-signature-v13/weights/best.pt")

# Load the image

image_path = "test4.jpg"  
image = cv2.imread(image_path)

# Perform inference
results = model(image)

# Get the output image with bounding boxes
annotated_image = results[0].plot()

# Display the result

cv2.namedWindow("YOLOv8 Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLOv8 Detection", 700, 700) 
cv2.imshow("YOLOv8 Detection", annotated_image)
cv2.waitKey(0)  # Press any key to close
cv2.destroyAllWindows()

# Save the output image
cv2.imwrite("output.jpg", annotated_image)  # Save the annotated image
print("Output saved as output.jpg")
