"""Speed vs accuracy of one detector in different deployment formats.

On a drone the model runs on small onboard hardware, so engineers trade
accuracy for frames per second. This script takes the VisDrone model from
../aerial-object-detection and measures every variant the same way:

  * PyTorch FP32 on GPU      - the training format
  * PyTorch FP16 on GPU      - half precision, usually ~same accuracy, faster
  * ONNX FP32 on CPU         - portable format, runs without PyTorch/GPU
  * ONNX INT8 on CPU         - 8-bit quantised, smaller and faster on CPU
  * (optional) TensorRT FP16 - NVIDIA's optimiser, used on Jetson boards

Accuracy = mAP50 on the VisDrone val split (Ultralytics validator, same for all),
speed = mean latency on the same 100 images at batch 1 after a warm-up.

Usage:
    python benchmark.py --weights ../aerial-object-detection/results/best.pt
"""
import argparse
import json
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
from ultralytics import YOLO, settings

HERE = Path(__file__).parent
OUT = HERE / "results"
DATA_ROOT = Path(os.environ.get("CV_DATA", Path.home() / "cv-data"))
VAL_IMAGES = DATA_ROOT / "datasets" / "VisDrone" / "images" / "val"


def latency_ms(model, images, device, half, imgsz):
    for img in images[:10]:  # warm-up: first runs include memory allocation, kernel tuning
        model.predict(img, device=device, half=half, imgsz=imgsz, verbose=False)
    times = []
    for img in images:
        t = time.perf_counter()
        model.predict(img, device=device, half=half, imgsz=imgsz, verbose=False)
        times.append(time.perf_counter() - t)
    return 1000 * float(np.mean(times))


def accuracy(model, device, half, imgsz):
    m = model.val(data="VisDrone.yaml", device=device, half=half, imgsz=imgsz, batch=1,
                  plots=False, verbose=False)
    return round(float(m.box.map50), 4), round(float(m.box.map), 4)


def quantize_int8(onnx_path):
    from onnxruntime.quantization import QuantType, quantize_dynamic
    out = onnx_path.with_name(onnx_path.stem + "_int8.onnx")
    quantize_dynamic(str(onnx_path), str(out), weight_type=QuantType.QUInt8)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(HERE.parent / "aerial-object-detection" / "results" / "best.pt"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--tensorrt", action="store_true")
    a = ap.parse_args()
    settings.update({"datasets_dir": str(DATA_ROOT / "datasets"), "runs_dir": str(DATA_ROOT / "runs")})

    images = sorted(str(p) for p in VAL_IMAGES.glob("*.jpg"))
    random.Random(0).shuffle(images)
    images = images[: a.n]
    # Export into an ASCII-only folder: ONNX shape inference on Windows cannot
    # open paths with non-Latin characters, and exports don't belong in the repo.
    export_dir = DATA_ROOT / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    weights = export_dir / "visdrone_yolo26n.pt"
    shutil.copy2(a.weights, weights)
    onnx_path = Path(YOLO(weights).export(format="onnx", imgsz=a.imgsz, simplify=True))
    variants = [
        ("PyTorch FP32", "GPU", weights, 0, False),
        ("PyTorch FP16", "GPU", weights, 0, True),
        ("ONNX FP32", "CPU", onnx_path, "cpu", False),
        ("ONNX INT8 (dynamic)", "CPU", quantize_int8(onnx_path), "cpu", False),
        ("PyTorch FP32", "CPU", weights, "cpu", False),
    ]
    if a.tensorrt:
        try:  # needs `pip install tensorrt` matching the CUDA version
            engine = Path(YOLO(weights).export(format="engine", imgsz=a.imgsz, half=True))
            variants.insert(2, ("TensorRT FP16", "GPU", engine, 0, True))
        except Exception as e:
            print(f"TensorRT skipped: {e}")

    rows = []
    for name, hw, path, device, half in variants:
        print(f"== {name} on {hw}")
        model = YOLO(str(path), task="detect")
        ms = latency_ms(model, images, device, half, a.imgsz)
        map50, map5095 = accuracy(model, device, half, a.imgsz)
        rows.append({"variant": name, "hardware": hw, "latency_ms": round(ms, 1), "fps": round(1000 / ms, 1),
                     "mAP50": map50, "mAP50-95": map5095, "file_mb": round(path.stat().st_size / 2**20, 1)})
        print(rows[-1])

    OUT.mkdir(exist_ok=True)
    (OUT / "benchmark.json").write_text(json.dumps(rows, indent=2))
    lines = ["| Variant | Hardware | Latency, ms | FPS | mAP50 | mAP50-95 | File, MB |", "|---|---|---|---|---|---|---|"]
    lines += [f"| {r['variant']} | {r['hardware']} | {r['latency_ms']} | {r['fps']} | {r['mAP50']} | {r['mAP50-95']} | {r['file_mb']} |" for r in rows]
    (OUT / "benchmark.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
