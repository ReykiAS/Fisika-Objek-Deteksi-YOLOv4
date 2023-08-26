# YOLOv4 Object Detection API Server

This repository is mixture version of https://github.com/theAIGuysCode/Object-Detection-API (YOLOv3 REST API) and
https://github.com/theAIGuysCode/tensorflow-yolov4-tflite (YOLOv4).
so that I added REST API functions on YOLOv4.

YOLOv4, YOLOv4-tiny Implemented in Tensorflow 2.0. Convert YOLO v4, YOLOv3, YOLO tiny .weights to .pb, .tflite and trt format for tensorflow, tensorflow lite, tensorRT, Flask, REST API.

## 1. Install
### Conda (Recommended)

```shell
# Tensorflow CPU
conda env create -f conda-cpu.yml
conda activate yolov4-cpu

# Tensorflow GPU
conda env create -f conda-gpu.yml
conda activate yolov4-gpu

# For apple silicon mac
conda env create -f conda-gpu-apple-silicon-mac.yml
conda activate yolov4-gpu
```

### Pip
```shell
# TensorFlow CPU
pip install -r requirements.txt

# TensorFlow GPU
pip install -r requirements-gpu.txt
```



### Detections by image files (POST http://localhost:5050/detections/by-image-files)
need more than 0 image files, multipart/form-data, key name is "images"

#### Request example(python.requests)
```python
import requests

url = "http://127.0.0.1:5050/detections/by-image-files"

payload={}
files=[
  ('images',('dog.jpg',open('/C:/Users/Qone/repos/YOLOv4-Object-Detection-API-Server/data/images/dog.jpg','rb'),'image/jpeg')),
  ('images',('kite.jpg',open('/C:/Users/Qone/repos/YOLOv4-Object-Detection-API-Server/data/images/kite.jpg','rb'),'image/jpeg'))
]
headers = {}

response = requests.request("POST", url, headers=headers, data=payload, files=files)

print(response.text)
```


<br><br>

## Reference
for more information
- https://github.com/theAIGuysCode/tensorflow-yolov4-tflite
- https://github.com/Qone2/YOLOv4-Object-Detection-API-Server/tree/main
- https://github.com/theAIGuysCode/Object-Detection-API
- https://www.youtube.com/channel/UCrydcKaojc44XnuXrfhlV8Q
