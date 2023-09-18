from openvino.runtime import Core, Model
from pathlib import Path
from typing import Tuple
from ultralytics.yolo.utils import ops
import torch
import numpy as np

from typing import Tuple, Dict
import cv2
import numpy as np
from PIL import Image
from ultralytics.yolo.utils.plotting import colors

from utils import *

##################################################
# inference
##################################################
label_map = {0: "face"}
IMAGE_PATH = Path('demo/dummy.jpg')
openvino_model_path = 'pretrained_models/face_detection/yolov8_openvino/best.xml'
device = 'CPU'
core = Core()

det_ov_model = core.read_model(openvino_model_path)
if device != "CPU":
    det_ov_model.reshape({0: [1, 3, 640, 640]})
det_compiled_model = core.compile_model(det_ov_model, device)

input_image = np.array(Image.open(IMAGE_PATH))
detections = detect(input_image, det_compiled_model)[0]
image_with_boxes = draw_results(detections, input_image, label_map)

Image.fromarray(image_with_boxes).show()

 
# Reading the video from a file
video = cv2.VideoCapture("demo/dataset_cam1.mp4")
 
# Checking whether the video has opened using the isOpened function
if (video.isOpened() == False):
    print("Error opening the video file")
 
# When the video has been opened successfully, we'll read each frame of the video using a loop
while(video.isOpened()):
    ret, frame = video.read()
    if ret == True:
        # inference
        input_image = np.array(frame)
        detections = detect(input_image, det_compiled_model)[0]
        image_with_boxes = draw_results(detections, input_image, label_map)

        cv2.imshow('Frame',image_with_boxes)
        # Using waitKey to display each frame of the video for 1 ms
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
 
cv2.destroyAllWindows()
