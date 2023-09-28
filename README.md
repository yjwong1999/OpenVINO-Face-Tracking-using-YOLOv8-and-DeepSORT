# OpenVINO-Face-Tracking-using-YOLOv8-and-DeepSORT

## Download the Repo

- Clone the repository
```
git clone https://github.com/yjwong1999/OpenVINO-Face-Tracking-using-YOLOv8-and-DeepSORT.git
```

- Goto cloned folder (we will do everything in this folder)
```
cd OpenVINO-Face-Tracking-using-YOLOv8-and-DeepSORT
```

## A. Train YOLOv8 Face Detection model -> Convert to OpenVINO

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
<click the Colab link below>
<run the codes step by step>
```
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1bnRcWCp1Y6Jf7l2NORiZ4pDLvgSmDmZw?usp=sharing) </br>


## B. Train OpenSphere Face Recognition model -> Convert to OpenVINO

- Refer [Training your OpenSphere Face Recognition Model using QMUL_SurvFace or any Custom Dataset](https://github.com/yjwong1999/opensphere)

## C. Multi Camera Face Detection and Tracking (MCFDR)

- Create a conda environment for MCFDR, and activate it
```
conda create --name mcfdr python=3.8.10
conda activate mcfdr 
```

- Goto yolo_tracking folder
```
cd yolo_tracking
```

- Install dependencies
```
# install pytorch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# install remaining requirements
pip3 install -r requirements.txt
```


## Known Issues
[issue](https://github.com/mikel-brostrom/yolo_tracking/issues/1071#issuecomment-1684865948)

Please uninstall your ultralytics, use the owner de version

Main difference between my yolo_tracking and the original is:</br> 
```yolo_tracking/boxmot/trackers/strongsort/sort/tracker.py```,</br>
where mine will perform cosine distance with id bank before assinging ID



