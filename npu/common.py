"""Shared pieces for the NPU examples: preprocessing, YOLO decoding, evaluation.

Both NPU toolchains (Rockchip RKNN, Hailo DFC) run on a Linux x86 PC, and
both ship a simulator / emulator of the chip. So accuracy of the quantised
model can be measured without the board; speed needs the real hardware.

Evaluation = the same simple metric as in aerial-object-detection/sahi_compare:
a detection is a hit if it overlaps a ground-truth box of the same class with
IoU >= 0.5 (confidence >= 0.25). Reported: precision, recall, recall on small
objects (< 32x32 px).
"""
import os
import random
from pathlib import Path

import cv2
import numpy as np

DATA_ROOT = Path(os.environ.get("CV_DATA", Path.home() / "cv-data"))
VISDRONE = DATA_ROOT / "datasets" / "VisDrone"
SMALL = 32 * 32
CONF, NMS_IOU, MATCH_IOU = 0.25, 0.7, 0.5


def sample_images(split, n, seed):
    paths = sorted(str(p) for p in (VISDRONE / "images" / split).glob("*.jpg"))
    random.Random(seed).shuffle(paths)
    return paths[:n]


def letterbox(img, size=640):
    """Keep aspect ratio, pad with grey 114 (same as Ultralytics). Returns image, scale, pad."""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = round(h * r), round(w * r)
    out = np.full((size, size, 3), 114, np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    out[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out, r, (left, top)


def decode(pred, r, pad):
    """YOLO output (14, 8400): 4 box values (cx, cy, w, h in input pixels) + 10 class scores.
    Returns boxes (x1, y1, x2, y2 in original image pixels), scores, classes after NMS."""
    pred = np.asarray(pred, dtype=np.float32)
    pred = pred.reshape(pred.shape[-2], pred.shape[-1])  # (1, 14, 8400) -> (14, 8400)
    if pred.shape[0] > pred.shape[1]:
        pred = pred.T
    return postprocess(pred[:4].T, pred[4:].T, r, pad)


def decode_raw(maps, r, pad, layout="nchw"):
    """Raw head outputs, as NPUs return them when the graph is cut at the head convs:
    per scale a box map (4 ch: left/top/right/bottom distances in grid cells, YOLO26 has
    no DFL) and a class map (10 ch, logits). Decoding runs on the host in float32."""
    maps = [np.asarray(m, np.float32).reshape(m.shape[-3:]) for m in maps]
    if layout == "nhwc":
        maps = [m.transpose(2, 0, 1) for m in maps]
    box_maps = sorted([m for m in maps if m.shape[0] == 4], key=lambda m: -m.shape[1])
    cls_maps = sorted([m for m in maps if m.shape[0] != 4], key=lambda m: -m.shape[1])
    size = 640
    boxes, scores = [], []
    for b, c in zip(box_maps, cls_maps):
        h, w = b.shape[1:]
        stride = size / h
        gy, gx = np.mgrid[0:h, 0:w] + 0.5  # anchor = cell centre
        l, t, rr, bb = b.reshape(4, -1)
        ax, ay = gx.ravel(), gy.ravel()
        x1, y1, x2, y2 = ax - l, ay - t, ax + rr, ay + bb
        boxes.append(np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], 1) * stride)
        scores.append(1 / (1 + np.exp(-c.reshape(c.shape[0], -1).T)))
    return postprocess(np.concatenate(boxes), np.concatenate(scores), r, pad)


def postprocess(boxes, cls_scores, r, pad):
    """boxes: (N, 4) cx, cy, w, h in input pixels; cls_scores: (N, classes) in 0..1."""
    cls = cls_scores.argmax(1)
    scores = cls_scores.max(1)
    keep = scores >= CONF
    boxes, scores, cls = boxes[keep], scores[keep], cls[keep]
    if len(boxes) == 0:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)
    xywh = boxes.copy()
    xywh[:, :2] -= xywh[:, 2:] / 2
    idx = cv2.dnn.NMSBoxesBatched(xywh.tolist(), scores.tolist(), cls.tolist(), CONF, NMS_IOU)
    idx = np.array(idx, dtype=int).reshape(-1)
    xyxy = np.concatenate([xywh[idx, :2], xywh[idx, :2] + xywh[idx, 2:]], 1)
    xyxy[:, [0, 2]] -= pad[0]
    xyxy[:, [1, 3]] -= pad[1]
    return xyxy / r, scores[idx], cls[idx]


def ground_truth(img_path, shape):
    label = Path(str(img_path).replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}")).with_suffix(".txt")
    h, w = shape[:2]
    if not label.exists():
        return np.zeros((0, 4)), np.zeros(0, int)
    a = np.loadtxt(label, ndmin=2)
    if a.size == 0:
        return np.zeros((0, 4)), np.zeros(0, int)
    cx, cy, bw, bh = a[:, 1] * w, a[:, 2] * h, a[:, 3] * w, a[:, 4] * h
    return np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1), a[:, 0].astype(int)


def iou(a, b):
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), 2)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None] - inter + 1e-9)


class Scorer:
    """Greedy same-class matching at IoU >= 0.5."""

    def __init__(self):
        self.tp = self.fp = self.gt = self.gt_small = self.tp_small = 0

    def add(self, boxes, cls, gt_boxes, gt_cls):
        self.gt += len(gt_boxes)
        small = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1]) < SMALL
        self.gt_small += int(small.sum())
        matched = np.zeros(len(gt_boxes), bool)
        if len(boxes) and len(gt_boxes):
            m = iou(boxes, gt_boxes)
            m[:, :] *= cls[:, None] == gt_cls[None, :]
            for i in range(len(boxes)):
                j = int(m[i].argmax())
                if m[i, j] >= MATCH_IOU and not matched[j]:
                    matched[j] = True
                    m[:, j] = 0
        self.tp += int(matched.sum())
        self.tp_small += int((matched & small).sum())
        self.fp += len(boxes) - int(matched.sum())

    def result(self):
        return {"precision": round(self.tp / max(self.tp + self.fp, 1), 4),
                "recall": round(self.tp / max(self.gt, 1), 4),
                "recall_small": round(self.tp_small / max(self.gt_small, 1), 4)}
