```bash
# torch FD + OpenVINO ReID + strongsort
python3 track.py --yolo-model ../pretrained_models/face_detection/yolov8_pytorch.pt --reid-model ../pretrained_models/face_recognition/sfnet20_openvino_model --tracking-method strongsort --source source.streams --device 0 --half --show

# torch FD + onnx FR + strongsort
python3 track.py --yolo-model ../pretrained_models/face_detection/yolov8_pytorch.pt --reid-model ../pretrained_models/face_recognition/backbone_10000.onnx --tracking-method strongsort --source source.streams --device 0 --half --show

# torch FD + onnx FR + deepocsort (default)
python3 track.py --yolo-model ../pretrained_models/face_detection/yolov8_pytorch.pt --reid-model ../pretrained_models/face_recognition/backbone_10000.onnx --tracking-method deepocsort --source source.streams --device 0 --half --show

# torch FD + onnx FR + ocsort
python3 track.py --yolo-model ../pretrained_models/face_detection/yolov8_pytorch.pt --reid-model ../pretrained_models/face_recognition/backbone_10000.onnx --tracking-method ocsort --source source.streams --device 0 --half --show

# torch FD + light ReID + deepocsort (default)
python3 track.py --yolo-model ../pretrained_models/face_detection/yolov8_pytorch.pt --reid-model lmbn_n_cuhk03_d.pt --source source.streams --device 0 --half --show

# torch FD + light ReID + botsort
python3 track.py --yolo-model ../pretrained_models/face_detection/yolov8_pytorch.pt --reid-model lmbn_n_cuhk03_d.pt --tracking-method botsort --source source.streams --device 0 --half --show
```
