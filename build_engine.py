from ultralytics import YOLO

# Load a YOLOv8n PyTorch model
model = YOLO("weights.pt")

# Export the model
model.export(format="engine")  # creates 'yolov8n.engine'

# Load the exported TensorRT model
trt_model = YOLO("weights.engine")

# Run inference
results = trt_model("https://ultralytics.com/images/bus.jpg") 

print(results[0].boxes)  # Print the results
print(results[0].boxes.xyxy)  # Print bounding box coordinates
print(results[0].boxes.cls)  # Print class labels
print(results[0].boxes.conf)  # Print confidence scores