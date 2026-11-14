"""Hailo-8L (Raspberry Pi AI HAT+, 13 TOPS): quantise and compile the YOLO
model with the Hailo Dataflow Compiler (DFC 3.x), and check its accuracy in
Hailo's emulator, without the board.

Same recipe as Ultralytics' `format="hailo"` export:
  * the ONNX graph is cut at the head's last convolutions (box + class map
    per scale); box decoding runs on the host CPU in float,
  * the six output layers stay at 16-bit (a16_w16): YOLO26 has no DFL, so its
    raw box distances need the wider range,
  * the rest of the network is INT8, calibrated on real VisDrone images.

Emulator runs: SDK_FP_OPTIMIZED (the graph Hailo will run, still float) and
SDK_QUANTIZED (bit-exact INT8 emulation of the chip). Then the HEF file for
the board is compiled.

Run inside Linux x86 (WSL2 is fine) with the DFC wheel from Hailo's
Developer Zone installed (needs a lot of RAM; 12 GB was enough for nano):
    python hailo8l.py --weights best.pt --level 1
On the board: copy the .hef, run it with HailoRT / Ultralytics.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

import common

WORK = Path.home() / "npu" / "hailo"
OUT = Path(__file__).resolve().parent.parent / "results"


def letterboxed_rgb(paths):
    return np.stack([common.letterbox(cv2.imread(p))[0][:, :, ::-1] for p in paths])


def evaluate(runner, context, images, batch=8):
    from hailo_sdk_client import InferenceContext

    score, t0 = common.Scorer(), time.perf_counter()
    with runner.infer_context(getattr(InferenceContext, context)) as ctx:
        for i in range(0, len(images), batch):
            paths = images[i:i + batch]
            originals = [cv2.imread(p) for p in paths]
            boxes_in = [common.letterbox(img) for img in originals]
            data = np.stack([b[0][:, :, ::-1] for b in boxes_in]).astype(np.float32)
            outs = runner.infer(ctx, data)
            outs = outs if isinstance(outs, list) else [outs]
            for j, (img, (_, r, pad)) in enumerate(zip(originals, boxes_in)):
                boxes, _, cls = common.decode_raw([o[j] for o in outs], r, pad, layout="nhwc")
                score.add(boxes, cls, *common.ground_truth(paths[j], img.shape))
    return {**score.result(), "emulator_s": round(time.perf_counter() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(Path.home() / "npu" / "visdrone_yolo26n.pt"))
    ap.add_argument("--n", type=int, default=100, help="validation images to evaluate")
    ap.add_argument("--calib", type=int, default=256, help="calibration images")
    ap.add_argument("--level", type=int, default=1, help="DFC optimization level (2 = Ultralytics default, needs GPU)")
    ap.add_argument("--arch", default="hailo8l")
    a = ap.parse_args()
    from hailo_sdk_client import ClientRunner

    WORK.mkdir(parents=True, exist_ok=True)
    w = WORK / "visdrone_yolo26n.pt"
    w.write_bytes(Path(a.weights).read_bytes())
    model = YOLO(str(w))
    head = len(model.model.model) - 1
    end_nodes = [f"/model.{head}/cv{b}.{i}/cv{b}.{i}.2/Conv" for b in (2, 3) for i in range(3)]
    # simplify=False: onnxslim would be auto-installed and pull numpy 2.x, which breaks the DFC (numpy 1.26)
    onnx_path = model.export(format="onnx", imgsz=640, opset=11, simplify=False)

    runner = ClientRunner(hw_arch=a.arch)
    runner.translate_onnx_model(str(onnx_path), "yolo26n_visdrone", end_node_names=end_nodes)
    outputs = ", ".join(f"output_layer{i + 1}" for i in range(len(end_nodes)))
    runner.load_model_script("\n".join([
        "normalization1 = normalization([0, 0, 0], [255, 255, 255])",
        f"model_optimization_flavor(optimization_level={a.level})",
        f"quantization_param([{outputs}], precision_mode=a16_w16)",
    ]))

    calib = letterboxed_rgb(common.sample_images("train", a.calib, seed=1)).astype(np.float32)
    t = time.perf_counter()
    runner.optimize(calib)
    optimize_s = round(time.perf_counter() - t, 1)
    del calib
    runner.save_har(str(WORK / f"yolo26n_visdrone_{a.arch}_quantized.har"))

    images = common.sample_images("val", a.n, seed=0)
    rows = []
    for label, context in [("Hailo emulator, float (SDK_FP_OPTIMIZED)", "SDK_FP_OPTIMIZED"),
                           ("Hailo emulator, INT8 + a16 outputs (SDK_QUANTIZED)", "SDK_QUANTIZED")]:
        rows.append({"model": label, **evaluate(runner, context, images)})
        print(rows[-1], flush=True)

    t = time.perf_counter()
    hef = WORK / f"yolo26n_visdrone_{a.arch}.hef"
    hef.write_bytes(runner.compile())
    summary = {"arch": a.arch, "optimization_level": a.level, "calibration_images": a.calib,
               "optimize_s": optimize_s, "compile_s": round(time.perf_counter() - t, 1),
               "hef_mb": round(hef.stat().st_size / 2**20, 1), "results": rows}
    OUT.mkdir(exist_ok=True)
    (OUT / f"{a.arch}_emulator.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
