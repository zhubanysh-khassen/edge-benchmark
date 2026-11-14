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
   - Dynamic quantisation is meant for Transformers and LSTMs (matmul-heavy). CNN detectors need **static** quantisation: calibrate activation ranges on 100–500 real images, then export a QDQ model, or let TensorRT/OpenVINO build an INT8 engine with a calibration set. **Part 2 below does exactly that.**

## Part 2: cheap hardware (CPU only, INT8)

Jetson boards are expensive, so many drone teams run the detector on a CPU (mini-PC, Raspberry Pi) or on a cheap NPU board (Rockchip RK3588, Hailo-8L). NPUs execute **only INT8** models, and INT8 only keeps accuracy with **static** quantisation calibrated on real images. [`cheap_hardware.py`](cheap_hardware.py) runs every variant on the CPU (AMD Ryzen 7 laptop, batch 1).
- **Model, ms:** the network's forward pass only.
- **End-to-end, ms:** adds JPEG decoding and pre/post-processing.

| Runtime | Precision | Input | Model, ms | End-to-end, ms | mAP50 | mAP50-95 | Size, MB |
|---|---|---|---|---|---|---|---|
| ONNX Runtime | FP32 | 640 | 24.4 | 32.2 | 0.291 | 0.162 | 9.4 |
| ONNX Runtime | INT8 static S8S8 | 640 | 64.9 | 72.9 | 0.275 | 0.150 | 3.0 |
| ONNX Runtime | INT8 static U8S8 | 640 | 19.8 | 28.0 | 0.275 | 0.150 | 3.0 |
| OpenVINO | FP32 | 640 | 20.4 | 27.4 | 0.291 | 0.162 | 9.6 |
| **OpenVINO** | **INT8 (NNCF)** | **640** | **9.2** | **15.8** | **0.289** | **0.160** | 3.2 |
| OpenVINO | INT8 (NNCF) | 480 | 6.3 | 12.1 | 0.225 | 0.119 | 3.1 |
| OpenVINO | INT8 (NNCF) | 320 | 4.0 | 9.0 | 0.138 | 0.070 | 3.1 |

### Findings

1. **The best CPU option is OpenVINO INT8.** It is 2.7× faster than ONNX FP32 (9.2 vs 24.4 ms) for −0.002 mAP50-95. That is about 60 FPS end-to-end on a laptop CPU, with no GPU and no NPU. OpenVINO runs on AMD as well as Intel x86 CPUs.
2. **INT8 has to match the hardware's integer instructions.** The same calibrated model is **3.3× slower** with signed activations (S8S8, 64.9 ms) than with unsigned ones (U8S8, 19.8 ms). The reason is that this CPU has no fast signed INT8 dot-product path in ONNX Runtime. Every target (x86, ARM, an NPU) has its own preferred INT8 scheme, so check it before quantising.
3. **The calibration method matters more than "INT8 or not".** My ONNX Runtime quantisation uses MinMax calibration: −0.012 mAP50-95. NNCF uses smarter range estimation and bias correction: only −0.002. In both cases the box-decoding part of the head is kept in float. That part mixes pixel coordinates (0–640) and scores (0–1), so one shared INT8 scale would destroy the boxes.
4. **Input size is the strongest speed knob, but small objects pay for it.** Going from 640 to 320 px makes the model 2.3× faster but halves mAP: a 20-px pedestrian becomes about 5 px. For drone footage, a smaller model at a high resolution beats a bigger model at a low one.

### NPU boards (RK3588, Hailo-8/8L)

The same INT8 pipeline applies to NPU boards, compiled by each vendor's toolchain (both are Linux x86-64 only):

```bash
# Rockchip RK3588 (Orange Pi 5, Radxa Rock 5B): rknn-toolkit2
yolo export model=best.pt format=rknn name=rk3588 quantize=8 data=VisDrone.yaml
# Hailo-8L (Raspberry Pi AI HAT+): Hailo Dataflow Compiler 3.x, ≥1024 calibration images recommended
yolo export model=best.pt format=hailo name=hailo8l data=VisDrone.yaml imgsz=640
```

## Run

```bash
pip install ultralytics onnx onnxruntime-gpu onnxslim openvino nncf
python benchmark.py                        # expects aerial-object-detection cloned next to this repo
python benchmark.py --weights path/to/best.pt
python benchmark.py --tensorrt             # adds a TensorRT FP16 engine (needs `pip install tensorrt`)
python cheap_hardware.py --weights path/to/best.pt   # part 2: CPU / INT8 / input sizes (~15 min)
```

Exported files (`.onnx`, `_int8.onnx`, `.engine`) go to a separate data folder, not the repo. Results are written to `results/benchmark.json` and `results/benchmark.md`.

> On Windows, ONNX shape inference cannot open paths with non-Latin characters, which is why the exports are written to an ASCII-only folder.

## Next steps

- **Measure on real NPUs:** compile for RK3588 and Hailo-8L, check accuracy in the vendors' simulators, then measure FPS on the boards.
- **Better calibration for ONNX Runtime:** percentile or entropy instead of MinMax, more calibration images.
- **TensorRT FP16/INT8:** the production format on NVIDIA Jetson.
