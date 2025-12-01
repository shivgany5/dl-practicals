# Object detection using YOLO and Pretrained Model

# Install YOLOv8
!pip install ultralytics

from ultralytics import YOLO

# Load pretrained YOLOv8 model
model = YOLO("yolov8n.pt")

# Path to your image
img_path = "sample.jpg"

# Run detection
results = model(img_path)

# Show results
results[0].show()

# Save output image
results[0].save("output.jpg")
