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
