"""Rockchip RK3588 NPU (Orange Pi 5, Radxa Rock 5B): build RKNN models and
check their accuracy in Rockchip's simulator, without the board.

  1. Ultralytics exports the trained YOLO to ONNX (opset 19, 640x640).
  2. rknn-toolkit2 (2.3.2: 2.5.0 fails on YOLO26) compiles it for rk3588:
     FP16, INT8 of the whole graph, INT8 with the graph cut at the head
     convolutions (box decoding on the CPU in float), and the same with
     INT8/FP16 hybrid layers. INT8 is calibrated on 200 real VisDrone
     training images (letterboxed exactly like at inference time, .npy, NCHW).
  3. Both models run in the simulator on 100 validation images; the ONNX
     FP32 model on the same images is the reference.

Run inside Linux x86 (WSL2 is fine) with rknn-toolkit2 installed:
    python rk3588.py --weights best.pt
The .rknn files go to ~/npu/rknn; copy one to the board and run it with
rknn-toolkit-lite2 (or export with Ultralytics: format=rknn name=rk3588).
"""
import argparse
import json
import re
import time
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from rknn.api import RKNN
from ultralytics import YOLO

import common

WORK = Path.home() / "npu" / "rknn"
OUT = Path(__file__).resolve().parent.parent / "results"


def calibration_set(n):
    folder = WORK / "calib"
    folder.mkdir(parents=True, exist_ok=True)
    lines = []
    for i, p in enumerate(common.sample_images("train", n, seed=1)):
        img, _, _ = common.letterbox(cv2.imread(p))
        f = folder / f"{i:04d}.npy"
        np.save(f, img[:, :, ::-1].transpose(2, 0, 1)[None])  # 1x3x640x640 (NCHW), RGB, uint8
        lines.append(str(f))
    txt = WORK / "calib.txt"
    txt.write_text("\n".join(lines) + "\n")
    return txt


def head_outputs(onnx_path):
    """Output tensors of the head's last convolutions (box + class map per scale).
    Cutting the NPU graph here keeps the box decoding (pixel-range anchor and
    stride math) out of INT8; it runs on the board's CPU in float instead."""
    graph = onnx.load(str(onnx_path)).graph
    head = "/model.%d/" % max(int(m.group(1)) for n in graph.node if (m := re.search(r"/model\.(\d+)/", n.name)))
    ops = {"Reshape", "Concat"}
    ends = {x for n in graph.node if n.op_type in ops for x in n.input}
    ends -= {x for n in graph.node if n.op_type not in ops for x in n.input}
    return [n.output[0] for n in graph.node if n.op_type == "Conv" and head in n.name and n.output[0] in ends]


def build(onnx_path, name, int8, calib_txt, cut_head=False, hybrid=False):
    rknn = RKNN(verbose=False)
    config = {"mean_values": [[0, 0, 0]], "std_values": [[255, 255, 255]], "target_platform": "rk3588"}
    if hybrid:  # keep layers whose INT8 output drifts (cosine < 0.999) in FP16
        config["auto_hybrid_cos_thresh"] = 0.999
    assert rknn.config(**config) == 0  # mean/std: the NPU does the 0..255 -> 0..1 scaling itself
    assert rknn.load_onnx(model=str(onnx_path), outputs=head_outputs(onnx_path) if cut_head else None) == 0
    kwargs = {"do_quantization": int8}
    if int8:
        kwargs.update(dataset=str(calib_txt), auto_hybrid=hybrid)
    assert rknn.build(**kwargs) == 0
    path = WORK / f"visdrone_yolo26n_{name}.rknn"
    assert rknn.export_rknn(str(path)) == 0
    assert rknn.init_runtime() == 0  # no target given -> simulator on this PC
    return rknn, path


def evaluate(run, images, decoder=common.decode):
    score, times = common.Scorer(), []
    for p in images:
        img = cv2.imread(p)
        lb, r, pad = common.letterbox(img)
        t = time.perf_counter()
        pred = run(lb[:, :, ::-1])
        times.append(time.perf_counter() - t)
        boxes, _, cls = decoder(pred, r, pad)
        score.add(boxes, cls, *common.ground_truth(p, img.shape))
    return {**score.result(), "sim_ms_per_image": round(1000 * float(np.mean(times)), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(Path.home() / "npu" / "visdrone_yolo26n.pt"))
    ap.add_argument("--n", type=int, default=100, help="validation images to evaluate")
    ap.add_argument("--calib", type=int, default=200, help="calibration images for INT8")
    a = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)

    w = WORK / "visdrone_yolo26n.pt"
    w.write_bytes(Path(a.weights).read_bytes())
    onnx_path = Path(YOLO(str(w)).export(format="onnx", imgsz=640, opset=19, simplify=True))
    images = common.sample_images("val", a.n, seed=0)
    calib_txt = calibration_set(a.calib)

    rows = []
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ref = lambda rgb: sess.run(None, {"images": (rgb.transpose(2, 0, 1)[None] / 255.0).astype(np.float32)})[0]
    rows.append({"model": "ONNX FP32 (reference, CPU)", **evaluate(ref, images)})
    print(rows[-1])

    variants = [  # label, file name, int8, cut head + decode on CPU, hybrid INT8/FP16
        ("RKNN FP16", "fp16", False, False, False),
        ("RKNN INT8, whole graph", "int8_full", True, False, False),
        ("RKNN INT8, head cut, decode on CPU", "int8_cut", True, True, False),
        ("RKNN INT8 + FP16 hybrid, head cut", "int8_hybrid_cut", True, True, True),
    ]
    for label, name, int8, cut, hybrid in variants:
        rknn, path = build(onnx_path, name, int8, calib_txt, cut, hybrid)
        if cut:
            run = lambda rgb: rknn.inference(inputs=[rgb[None]], data_format="nhwc")
            decoder = lambda pred, r, pad: common.decode_raw(pred, r, pad, layout="nchw")
        else:
            run = lambda rgb: rknn.inference(inputs=[rgb[None]], data_format="nhwc")[0]
            decoder = common.decode
        rows.append({"model": label, **evaluate(run, images, decoder),
                     "file_mb": round(path.stat().st_size / 2**20, 1)})
        rknn.release()
        print(rows[-1], flush=True)

    OUT.mkdir(exist_ok=True)
    (OUT / "rk3588_simulator.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
