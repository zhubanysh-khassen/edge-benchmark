# Edge Benchmark: one detector, five deployment formats

On a drone the detector runs on small on-board hardware (Jetson, a mini-PC, sometimes plain CPU). The engineering question is how much accuracy is lost for each extra frame per second. This project takes the YOLO26-nano model trained on VisDrone in [aerial-object-detection](https://github.com/zhubanysh-khassen/aerial-object-detection) and measures every variant the same way.

- **Speed:** mean end-to-end latency over 100 validation images at batch 1, after 10 warm-up runs. "End-to-end" includes JPEG decoding, resizing, the network and NMS.
- **Accuracy:** mAP on the full VisDrone val split (548 images), using the same Ultralytics validator for every variant.

## Results (RTX 3050 Laptop 4 GB / Ryzen 7 laptop CPU, 640 px)

| Variant | Hardware | Latency, ms | FPS | mAP50 | mAP50-95 | File, MB |
|---|---|---|---|---|---|---|
| PyTorch FP32 | GPU | 23.7 | 42.1 | 0.290 | 0.161 | 5.2 |
| PyTorch FP16 | GPU | **20.5** | **48.7** | 0.290 | 0.160 | 5.2 |
| ONNX FP32 | CPU | 30.3 | 33.0 | **0.291** | **0.162** | 9.4 |
| ONNX INT8 (dynamic) | CPU | 56.7 | 17.6 | 0.271 | 0.143 | **2.7** |
| PyTorch FP32 | CPU | 32.8 | 30.5 | 0.290 | 0.161 | 5.2 |

## What the numbers say

1. **FP16 is free accuracy-wise.** mAP is unchanged to the third decimal and the model runs 13 % faster. On GPUs with tensor cores (every Jetson) half precision is the default first step.
2. **The network itself is not the bottleneck on GPU.** The model's own forward pass takes about 3 ms, while end-to-end time is about 20 ms. Most of the time goes to decoding a 2000×1500 JPEG and resizing it. In a real pipeline, frames come straight from the camera as raw arrays and preprocessing moves to the GPU, so measure end-to-end and not only `inference`.
3. **A nano model is already real-time on a laptop CPU.** ONNX Runtime on CPU gives 33 FPS with no loss in accuracy. ONNX is also the portable format: it runs without PyTorch, on ARM, on Windows or Linux.
4. **INT8 quantisation went wrong, and that is the useful lesson.** *Dynamic* quantisation made the file 3.5× smaller but the model **1.9× slower** and cost 0.019 mAP50-95.
   - Why: dynamic quantisation converts only the weights and computes activation scales at run time.
   - For convolutions, ONNX Runtime on x86 then falls back to generic `ConvInteger` kernels plus extra quantise/dequantise steps.
   - Dynamic quantisation is meant for Transformers and LSTMs (matmul-heavy). CNN detectors need **static** quantisation: calibrate activation ranges on 100–500 real images, then export a QDQ model, or let TensorRT/OpenVINO build an INT8 engine with a calibration set.

## Run

```bash
pip install ultralytics onnx onnxruntime-gpu onnxslim
python benchmark.py                        # expects aerial-object-detection cloned next to this repo
python benchmark.py --weights path/to/best.pt
python benchmark.py --tensorrt             # adds a TensorRT FP16 engine (needs `pip install tensorrt`)
```

Exported files (`.onnx`, `_int8.onnx`, `.engine`) go to a separate data folder, not the repo. Results are written to `results/benchmark.json` and `results/benchmark.md`.

> On Windows, ONNX shape inference cannot open paths with non-Latin characters, which is why the exports are written to an ASCII-only folder.

## Next steps

- **TensorRT FP16/INT8:** the production format on NVIDIA Jetson; typically 2–4× faster than PyTorch on the same GPU.
- **Static INT8** (`onnxruntime.quantization.quantize_static` with a calibration reader) and a comparison with OpenVINO on CPU.
- **Input size sweep (320 / 480 / 640 / 960):** for small drone targets, resolution usually matters more than precision format.
