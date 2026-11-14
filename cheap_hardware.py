"""Deployment on cheap hardware: CPU-only and INT8, the way NPU boards need it.

Many drone teams avoid Jetson because of its price and run the detector on a
CPU (mini-PC, Raspberry Pi) or on a cheap NPU board (Rockchip RK3588,
Hailo-8L). NPUs execute only INT8 models, and INT8 only works well with
*static* quantisation: activation ranges are calibrated on real images.

This script compares, on CPU at batch 1:
  * ONNX FP32                      - baseline
  * ONNX INT8 static (QDQ)         - calibrated on 200 VisDrone train images,
                                     signed (S8S8) vs unsigned (U8S8) activations
  * OpenVINO FP32 / INT8 (NNCF)    - Intel's CPU runtime, also runs on AMD
  * OpenVINO INT8 at 320/480/640   - input size: the first knob on weak hardware

Latency = model-only inference time (Ultralytics speed['inference']) and
end-to-end time incl. JPEG decoding and pre/post-processing.
Accuracy = mAP on the full VisDrone val split.

Usage:
    python cheap_hardware.py --weights path/to/best.pt
"""
import argparse
import json
import os
import random
import re
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import onnx
from ultralytics import YOLO, settings

HERE = Path(__file__).parent
OUT = HERE / "results"
DATA_ROOT = Path(os.environ.get("CV_DATA", Path.home() / "cv-data"))
IMAGES = DATA_ROOT / "datasets" / "VisDrone" / "images"
EXPORTS = DATA_ROOT / "exports" / "cheap"  # ASCII-only path: ONNX tools fail on non-Latin paths


def letterbox(img, size):
    """Same preprocessing as Ultralytics: keep aspect ratio, pad with grey 114."""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = round(h * r), round(w * r)
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = np.full((size, size, 3), 114, np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    out[top:top + nh, left:left + nw] = img
    return out


def calibration_batches(n, size):
    """n real training images as network input tensors (1x3xHxW, RGB, 0..1)."""
    paths = sorted((IMAGES / "train").glob("*.jpg"))
    random.Random(1).shuffle(paths)
    for p in paths[:n]:
        x = letterbox(cv2.imread(str(p)), size)[:, :, ::-1].transpose(2, 0, 1)
        yield np.ascontiguousarray(x, dtype=np.float32)[None] / 255.0


def quantize_onnx_static(fp32_path, size, n_calib, activations="u8"):
    """activations: 'u8' (U8S8) or 's8' (S8S8). On x86 CPUs without fast signed
    INT8 dot products, ONNX Runtime runs S8S8 much slower than U8S8."""
    from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat,
                                          QuantType, quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process

    model = onnx.load(str(fp32_path))
    input_name = model.graph.input[0].name
    # Keep the box-decoding part of the detection head in float: it mixes pixel
    # coordinates (0..640) and class scores (0..1) in one tensor, and a single
    # INT8 scale for both destroys the boxes. The head's convolutions stay INT8.
    idx = [int(m.group(1)) for n in model.graph.node if (m := re.search(r"/model\.(\d+)/", n.name))]
    head = f"/model.{max(idx)}/"
    exclude = [n.name for n in model.graph.node if head in n.name and n.op_type != "Conv"]

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = ({input_name: x} for x in calibration_batches(n_calib, size))

        def get_next(self):
            return next(self.it, None)

    pre = fp32_path.with_name(fp32_path.stem + "_pre.onnx")
    quant_pre_process(str(fp32_path), str(pre))
    out = fp32_path.with_name(fp32_path.stem + f"_int8_{activations}s8.onnx")
    act = QuantType.QUInt8 if activations == "u8" else QuantType.QInt8
    quantize_static(str(pre), str(out), Reader(), quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=act, weight_type=QuantType.QInt8,
                    calibrate_method=CalibrationMethod.MinMax, nodes_to_exclude=exclude)
    return out


def measure(path, images, imgsz):
    model = YOLO(str(path), task="detect")
    for img in images[:10]:  # warm-up
        model.predict(img, device="cpu", imgsz=imgsz, verbose=False)
    infer, total = [], []
    for img in images:
        t = time.perf_counter()
        r = model.predict(img, device="cpu", imgsz=imgsz, verbose=False)[0]
        total.append(time.perf_counter() - t)
        infer.append(r.speed["inference"])
    m = model.val(data="VisDrone.yaml", device="cpu", imgsz=imgsz, batch=1, plots=False, verbose=False)
    return {"model_ms": round(float(np.mean(infer)), 1), "end_to_end_ms": round(1000 * float(np.mean(total)), 1),
            "mAP50": round(float(m.box.map50), 4), "mAP50-95": round(float(m.box.map), 4)}


def size_mb(path):
    path = Path(path)
    files = path.rglob("*") if path.is_dir() else [path]
    return round(sum(f.stat().st_size for f in files if f.is_file()) / 2**20, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(HERE.parent / "aerial-object-detection" / "results" / "best.pt"))
    ap.add_argument("--n", type=int, default=100, help="images for the latency measurement")
    ap.add_argument("--calib", type=int, default=200, help="images for INT8 calibration")
    ap.add_argument("--sizes", default="320,480,640")
    a = ap.parse_args()
    settings.update({"datasets_dir": str(DATA_ROOT / "datasets"), "runs_dir": str(DATA_ROOT / "runs")})
    EXPORTS.mkdir(parents=True, exist_ok=True)

    images = sorted(str(p) for p in (IMAGES / "val").glob("*.jpg"))
    random.Random(0).shuffle(images)
    images = images[: a.n]

    def weights_for(size):  # one copy per size: exports are written next to the weights
        w = EXPORTS / f"visdrone_yolo26n_{size}.pt"
        shutil.copy2(a.weights, w)
        return w

    variants = []
    w640 = weights_for(640)
    onnx_fp32 = Path(YOLO(w640).export(format="onnx", imgsz=640, simplify=True))
    variants.append(("ONNX Runtime", "FP32", 640, onnx_fp32))
    variants.append(("ONNX Runtime", "INT8 static S8S8", 640, quantize_onnx_static(onnx_fp32, 640, a.calib, "s8")))
    variants.append(("ONNX Runtime", "INT8 static U8S8", 640, quantize_onnx_static(onnx_fp32, 640, a.calib, "u8")))
    variants.append(("OpenVINO", "FP32", 640, Path(YOLO(w640).export(format="openvino", imgsz=640))))
    for s in [int(v) for v in a.sizes.split(",")]:
        w = w640 if s == 640 else weights_for(s)
        ov_int8 = YOLO(w).export(format="openvino", imgsz=s, int8=True, data="VisDrone.yaml", fraction=0.05)
        variants.append(("OpenVINO", "INT8 (NNCF)", s, Path(ov_int8)))

    rows = []
    for runtime, precision, s, path in variants:
        print(f"== {runtime} {precision} {s}px")
        row = {"runtime": runtime, "precision": precision, "imgsz": s, **measure(path, images, s), "size_mb": size_mb(path)}
        rows.append(row)
        print(row)

    OUT.mkdir(exist_ok=True)
    (OUT / "cheap_hardware.json").write_text(json.dumps(rows, indent=2))
    lines = ["| Runtime | Precision | Input | Model, ms | End-to-end, ms | mAP50 | mAP50-95 | Size, MB |",
             "|---|---|---|---|---|---|---|---|"]
    lines += [f"| {r['runtime']} | {r['precision']} | {r['imgsz']} | {r['model_ms']} | {r['end_to_end_ms']} | "
              f"{r['mAP50']} | {r['mAP50-95']} | {r['size_mb']} |" for r in rows]
    (OUT / "cheap_hardware.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
