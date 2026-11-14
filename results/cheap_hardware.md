| Runtime | Precision | Input | Model, ms | End-to-end, ms | mAP50 | mAP50-95 | Size, MB |
|---|---|---|---|---|---|---|---|
| ONNX Runtime | FP32 | 640 | 24.4 | 32.2 | 0.2909 | 0.1621 | 9.4 |
| ONNX Runtime | INT8 static S8S8 | 640 | 64.9 | 72.9 | 0.275 | 0.1504 | 3.0 |
| ONNX Runtime | INT8 static U8S8 | 640 | 19.8 | 28.0 | 0.2747 | 0.1495 | 3.0 |
| OpenVINO | FP32 | 640 | 20.4 | 27.4 | 0.2909 | 0.1621 | 9.6 |
| OpenVINO | INT8 (NNCF) | 320 | 4.0 | 9.0 | 0.1376 | 0.0697 | 3.1 |
| OpenVINO | INT8 (NNCF) | 480 | 6.3 | 12.1 | 0.2246 | 0.1192 | 3.1 |
| OpenVINO | INT8 (NNCF) | 640 | 9.2 | 15.8 | 0.2891 | 0.1601 | 3.2 |
