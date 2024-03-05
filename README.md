# OpenVINO-Face-Tracking-using-YOLOv8-and-DeepSORT 

### (We are testing out the system in our testbed 🛠️ Please starred the repo for more updates)
### The face recognition module will be re-added soon, fixing some minor bugs :)

## Download the Repo

- Clone the repository
```
git clone https://github.com/yjwong1999/OpenVINO-Face-Tracking-using-YOLOv8-and-DeepSORT.git
```

- Goto cloned folder (we will do everything in this folder)
```
cd OpenVINO-Face-Tracking-using-YOLOv8-and-DeepSORT
```

### A. Multi Camera Face Detection and Tracking (MCFDR)

- Create a conda environment for MCFDR, and activate it
```
conda create --name mcfdr python=3.8.10
conda activate mcfdr 
```

- Install git
```
sudo apt install git
```

- Goto yolo_tracking folder
```
cd yolo_tracking
```

- Install dependencies
```
# install pytorch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# google api for drive upload
pip3 install google-api-python-client==2.111.0
pip3 install gspread==5.12.2
pip3 install oauth2client==4.1.3

# install remaining requirements
pip3 install -r requirements.txt
pip install gdown==4.6.1
```

- Find physical port(s) of the device which is connected to camera
```
# find physical port connected to our device
python3 find_port.py
```

- List down your video stream source(s) line by line
```
# inside source.streams
source 1
source 2
source 3
...
```

- Start deploying
```
# tracking without geofencing
python3 track_ori.py \
--yolo-model ../pretrained_models/face_detection/yolov8_openvino_model \
--reid-model ../pretrained_models/face_recognition/sfnet20_openvino_model \
--tracking-method strongsort \
--source source.streams \
--device 0 \
--half \
--save \
--save-id-crops

# tracking with geofencing
python3 track.py \
--yolo-model ../pretrained_models/face_detection/yolov8_openvino_model \
--reid-model ../pretrained_models/face_recognition/sfnet20_openvino_model \
--tracking-method strongsort \
--source source.streams \
--device 0 \
--half \
--save \
--save-id-crops \
--roi-xyxys "[0.3,0.0,0.6,0.5]"

# note that --roi-xyxys is/are the geofencing area [x1,y1,x2,y2], normalized to range 0 to 1
# for single video streams  : --roi-xyxys "[x1,y1,x2,y2]"
# for multiple video streams: --roi-xyxys "[x1,y1,x2,y2][x1,y1,x2,y2]..."
```

<details><summary>Retrain YOLOv8 & OpenSphere</summary>

### B. Train YOLOv8 Face Detection model -> Convert to OpenVINO
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1bnRcWCp1Y6Jf7l2NORiZ4pDLvgSmDmZw?usp=sharing) </br>

- Create a conda environment for YOLOv8, and activate it
```
conda create --name yolov8 python=3.8.10
conda activate yolov8 
```

- Install the dependencies
```
pip3 install jupyter
pip3 install -r requirements.txt
```

- Method 1: Jupyter notebook
```
jupyter notebook
<click the YOLOv8_Training_for_UFDD.ipynb>
<run the codes step by step>
```

- Method 2: Colab
  
Notice:
Please download the latest Jupyter Notebook (YOLOv8 Training for UFDD.ipynb.ipynb) from the provided Colab link.
```
<click the Colab link above>
<run the codes step by step>
```

### C. Train OpenSphere Face Recognition model -> Convert to OpenVINO

- Refer [Training your OpenSphere Face Recognition Model using QMUL_SurvFace or any Custom Dataset](https://github.com/yjwong1999/opensphere)

</details>


## Known Issues
[issue](https://github.com/mikel-brostrom/yolo_tracking/issues/1071#issuecomment-1684865948)

Please uninstall your ultralytics, the appropriate version will be downloaded automatically.</br>
Then:
```
pip install --no-cache "ultralytics @ git+https://github.com/mikel-brostrom/ultralytics.git"
```



## Acknowledgement
This work was supported by Greatech Integration (M) Sdn Bhd with project number 8084-0008.

### Reference Code
1. [Yolov8](https://github.com/ultralytics/ultralytics) </br>
2. [OpenSphere Face Recognition](https://github.com/ydwen/opensphere) </br>
3. [Yolov8 + DeepSort for Object Detection and Tracking](https://github.com/mikel-brostrom/yolov8_tracking) </br>
4. [mikel-brostrom's ultralytic, which is used in ref work 3](https://github.com/mikel-brostrom/ultralytics)
5. [How to list available cameras OpenCV/Python](https://stackoverflow.com/a/62639343)
6. [How to wget files from Google Drive](https://bcrf.biochem.wisc.edu/2021/02/05/download-google-drive-files-using-wget/)

- I recommend to use ```ultralytics==8.0.146``` to train your YOLOv8 model for this repo, since ref work [3] and [4] are modified based this version </br>
- OpenSphere is used to train Face Recognition model </br>
- This repo is heavily based on [3], with minor modifications </br>

### Difference between my module and reference code

1. Main difference between my [yolo_tracking](https://github.com/yjwong1999/OpenVINO-Face-Tracking-using-YOLOv8-and-DeepSORT/tree/main/yolo_tracking) module and the [original](https://github.com/mikel-brostrom/yolov8_tracking) is: ```yolo_tracking/boxmot/trackers/strongsort/sort/tracker.py```, where [mine](https://github.com/yjwong1999/OpenVINO-Face-Tracking-using-YOLOv8-and-DeepSORT/blob/main/yolo_tracking/boxmot/trackers/strongsort/sort/tracker.py) will perform cosine distance with id bank before assinging ID

2. Instead of using ```pip install ultralytics``` from the [original repo](https://github.com/ultralytics/ultralytics), I use my [modified ultralytics](https://github.com/yjwong1999/OpenVINO-Face-Tracking-using-YOLOv8-and-DeepSORT/tree/main/yolo_tracking/ultralytics) which is placed in: ```yolo_tracking/ultralytics```. My version is based on ref work [4], where I modified ```ultralytics/ultralytics/data/loaders.py``` to solve [this issue](https://github.com/ultralytics/ultralytics/issues/4493#issuecomment-1692142970)

## Cite this repository

```
@INPROCEEDINGS{10174362,
  author={Wong, Yi Jie and Huang Lee, Kian and Tham, Mau-Luen and Kwan, Ban-Hoe},
  booktitle={2023 IEEE World AI IoT Congress (AIIoT)}, 
  title={Multi-Camera Face Detection and Recognition in Unconstrained Environment}, 
  year={2023},
  volume={},
  number={},
  pages={0548-0553},
  doi={10.1109/AIIoT58121.2023.10174362}}
```

