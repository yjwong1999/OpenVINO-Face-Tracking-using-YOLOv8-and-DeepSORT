# Known Issues and How to Fix Them

This is a compiled list of potential issues that you may encounter. If your error/issue does not appear here, please refer to the official [yolo_tracking repo](https://github.com/mikel-brostrom/yolo_tracking). If problem still persist, feel free to create an issue in this repo.

### 1. Unsupported 'dets' input type '<class 'ultralytics.engine.results.Boxes'>', valid format is np.ndarray
Solution: [please uninstall ultralytics that you have manually downloaded](https://github.com/mikel-brostrom/yolo_tracking/issues/1071#issuecomment-1684865948)
```
pip uninstall ultralytics
```

### 2. Any ultralytics module issue
Solution: [please uninstall ultralytics that you have manually downloaded](https://github.com/mikel-brostrom/yolo_tracking/issues/1071#issuecomment-1684865948)
```
pip uninstall ultralytics
```

### 3. AttributeError: 'NoneType' object has no attribute 'groups'
Solution: [gdown version](https://github.com/mikel-brostrom/yolo_tracking/issues/1248#issuecomment-1889563576)
```
pip install gdown==4.6.1
```

### 4. onnxruntime.capi.onnxruntime_pybind11_state.InvalidProtobuf: [ONNXRuntimeError] : 7 : INVALID_PROTOBUF
Solution: [onnx and onnx runtime version should match](https://onnxruntime.ai/docs/reference/compatibility.html) </br>
Note that you may need to check protobuf version also
```
(EXAMPLE)
# Dependencies for onnx (usually i will do this in a separate conda env)
pip install onnx==1.14.1
pip install protobuf==4.24.4
pip install typing-extensions==4.7.1

# Dependencies for onnx runtime
pip install onnxruntime=1.16
```

### 5. How to export custom ReID models (.pt format) to TorchScript, ONNX, OpenVINO and TensorRT
Solution: Use the following [code](https://github.com/mikel-brostrom/yolo_tracking/wiki/ReID-multi-framework-model-export). For more arguments details, refer the [source code](https://github.com/yjwong1999/OpenVINO-Face-Tracking-using-YOLOv8-and-DeepSORT/blob/main/yolo_tracking/boxmot/appearance/reid_export.py)
```
cd yolo_tracking
python3 boxmot/appearance/reid_export.py --weights <path/to/your/model.pt> --include torchscript onnx openvino engine --device 0 --batch-size <max_num_expected_objects> --dynamic
```

### 6. How to export OpenSphere model into ONNX, OpenVINO format
Solution: Refer OpenSphere [export guide](https://github.com/yjwong1999/opensphere/blob/main/README.md#export-opensphere-model-to-other-format-for-future-usage)
```
# make sure you do it on another conda env, to prevent version conflict issue
```

### 7. onnxruntime_inference_collection.py:69: UserWarning: Specified provider 'CUDAExecutionProvider' is not in available provider names.Available providers: 'AzureExecutionProvider, CPUExecutionProvider'
Solution: You may need to reinstall onnxruntime and onnxruntime-gpu version, to avoid conflicts between onnxruntime and onnxruntime-gpu [link 1](https://stackoverflow.com/a/76463621)[link 2](https://huggingface.co/docs/optimum/main/en/onnxruntime/usage_guides/gpu#cuda-installation). You can check [here](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirementsto)
```
pip uninstall onnxruntime
pip uninstall onnxruntime-gpu
pip install optimum[onnxruntime-gpu]==1.16

# disclaimer: please select the appropriate version based on your os/pytorch/etc...
```

### 8. segmentation fault (core dumped)
Solution: There is too many possibility which contributes to this error, making it hard to pinpoint the exact root cause for this error. However, there are a few possibilities
```
1. pytorch version you install is not compatible with the CUDA and NVIDIA driver version.
2. you passed the arguments wrongly
```
