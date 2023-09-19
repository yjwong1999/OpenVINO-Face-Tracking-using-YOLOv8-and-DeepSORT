# OpenVINO-Face-Tracking-using-YOLOv8-and-DeepSORT

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1bnRcWCp1Y6Jf7l2NORiZ4pDLvgSmDmZw?usp=sharing) </br>
Train YOLOv8 using UFDD dataset + Convert to OpenVINO

Notice:
Please download the latest Jupyter Notebook (YOLOv8 Training for UFDD.ipynb.ipynb) from the provided Colab link.

## Setup the codes/environment

- Create a conda environment, and activate it
```
conda create --name mcfdr python=3.8.10
conda activate mcfdr 
```


- Clone the repository
```
git clone https://github.com/yjwong1999/OpenVINO-Face-Tracking-using-YOLOv8-and-DeepSORT.git
```

- Goto cloned folder
```
cd OpenVINO-Face-Tracking-using-YOLOv8-and-DeepSORT
```

- Install the dependencies
```
pip install jupyter
pip install -r requirements.txt
```

## Train YOLOv8 Face Detection model -> Convert to OpenVINO

- Make sure you activate the conda environment
```
conda activate mcfdr 
```

- Method 1: Jupyter notebook
```
jupyter notebook
<click the YOLOv8_Training_for_UFDD.ipynb>
<run the codes step by step>
```

- Method 2: Colab
```
<click the Colab link above>
<run the codes step by step>
```

## Train OpenSphere Face Recognition model -> Convert to OpenVINO

- Refer [Training your OpenSphere Face Recognition Model using QMUL_SurvFace or any Custom Dataset](https://github.com/yjwong1999/opensphere)



