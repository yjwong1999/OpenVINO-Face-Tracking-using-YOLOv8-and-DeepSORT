from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  # load an official model
model = YOLO('../pretrained_models/face_detection/yolov8_pytorch.pt')  # load a custom model

# Predict with the model
results = model('https://www.youtube.com/watch?v=PCdR65T4Yq8', show=True)  # predict on an image
