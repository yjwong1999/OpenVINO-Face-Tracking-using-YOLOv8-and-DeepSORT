from ultralytics import YOLO

# Load a model
model = YOLO('../pretrained_models/face_detection/yolov8_pytorch.pt')  # load a custom model

# Predict with the model
results = model('source.streams', show=True, vid_stride=2, device=0)  
