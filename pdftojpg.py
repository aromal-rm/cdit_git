from pdf2image import convert_from_path
import os

# Input PDF file
pdf_path = "D:\\Projects\\cdit\\YOLO\\test4.pdf"  

# Output folder
output_folder = "output_images"
os.makedirs(output_folder, exist_ok=True)

# Convert PDF to images
images = convert_from_path(pdf_path, dpi=300)  # Higher dpi = better quality

# Save each page as JPG
for i, image in enumerate(images):
    image_path = os.path.join(output_folder, f"page_{i + 1}.jpg")
    image.save(image_path, "JPEG")
    print(f"Saved: {image_path}")

print("PDF conversion completed!")
