| Variant | Hardware | Latency, ms | FPS | mAP50 | mAP50-95 | File, MB |
|---|---|---|---|---|---|---|
| PyTorch FP32 | GPU | 23.7 | 42.1 | 0.2895 | 0.161 | 5.2 |
| PyTorch FP16 | GPU | 20.5 | 48.7 | 0.2896 | 0.1602 | 5.2 |
| ONNX FP32 | CPU | 30.3 | 33.0 | 0.2909 | 0.1621 | 9.4 |
| ONNX INT8 (dynamic) | CPU | 56.7 | 17.6 | 0.2713 | 0.1429 | 2.7 |
| PyTorch FP32 | CPU | 32.8 | 30.5 | 0.2895 | 0.161 | 5.2 |
