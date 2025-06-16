# 🔍 Seal & Signature Detection System

<div align="center">

![YOLO](https://img.shields.io/badge/YOLO-v8-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=OpenCV&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

**🎯 YOLOv8-based deep learning solution for automated seal and signature detection in documents**

[️ Installation](#installation) • [📖 Usage](#usage)

---

</div>

## 🌟 Overview

This project implements a **YOLOv8-based detection system** for identifying and locating seals and signatures in documents. The model can detect two classes: **seals** and **signatures** with high accuracy.

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- GPU with CUDA support (optional, for faster processing)

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/seal-signature-detection.git
cd seal-signature-detection

# Install required packages
pip install -r requirements.txt
```

### Required Dependencies

The main dependencies include:
- `ultralytics>=8.0.0` - YOLOv8 framework
- `torch>=1.9.0` - PyTorch deep learning framework
- `opencv-python>=4.5.0` - Computer vision library
- `pdf2image>=2.1.0` - PDF processing utility
- `numpy>=1.21.0` - Numerical computing
- `Pillow>=8.3.0` - Image processing

## 📖 Usage

### 🖼️ Basic Image Detection

```python
from ultralytics import YOLO
import cv2

# Load the trained model
model = YOLO("Models/runs/seal-signature-v13/weights/best.pt")

# Load and process an image
image_path = "your_document.jpg"
results = model(image_path)

# Display results with bounding boxes
annotated_image = results[0].plot()
cv2.imshow("Seal & Signature Detection", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the annotated result
cv2.imwrite("detection_result.jpg", annotated_image)
```

### 📄 PDF Document Processing

```python
from pdf2image import convert_from_path
from ultralytics import YOLO
import cv2

# Convert PDF to images
pdf_path = "document.pdf"
images = convert_from_path(pdf_path, dpi=300)

# Load the model
model = YOLO("Models/runs/seal-signature-v13/weights/best.pt")

# Process each page
for i, image in enumerate(images):
    # Save page as image first
    page_path = f"page_{i+1}.jpg"
    image.save(page_path, "JPEG")
    
    # Run detection
    results = model(page_path)
    annotated_image = results[0].plot()
    
    # Save result
    cv2.imwrite(f"page_{i+1}_detected.jpg", annotated_image)
    print(f"Processed page {i+1}")
```

### 🔄 Batch Processing Multiple Files

```python
from ultralytics import YOLO
import os

# Load model
model = YOLO("Models/runs/seal-signature-v13/weights/best.pt")

# Process all images in a folder
input_folder = "input_images"
output_folder = "output_results"
os.makedirs(output_folder, exist_ok=True)

# Supported image formats
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

for filename in os.listdir(input_folder):
    if any(filename.lower().endswith(ext) for ext in image_extensions):
        image_path = os.path.join(input_folder, filename)
        
        # Run detection
        results = model(image_path)
        
        # Save annotated result
        output_path = os.path.join(output_folder, f"detected_{filename}")
        results[0].save(output_path)
        
        print(f"Processed: {filename}")
```

### 🎯 Getting Detection Results

```python
from ultralytics import YOLO

# Load model
model = YOLO("Models/runs/seal-signature-v13/weights/best.pt")

# Run detection
results = model("document.jpg")

# Access detection data
for result in results:
    # Get bounding boxes
    boxes = result.boxes
    
    if boxes is not None:
        for box in boxes:
            # Get coordinates
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            # Get confidence score
            confidence = box.conf[0].item()
            
            # Get class (0: seal, 1: signature)
            class_id = int(box.cls[0].item())
            class_names = ['seal', 'signature']
            class_name = class_names[class_id]
            
            print(f"Detected {class_name} with confidence {confidence:.2f}")
            print(f"Location: ({x1:.0f}, {y1:.0f}) to ({x2:.0f}, {y2:.0f})")
```

---

<div align="center">

**⭐ Star this repository if you found it helpful!**

**Contact**: your.email@example.com

Made with ❤️ and Python

</div>
