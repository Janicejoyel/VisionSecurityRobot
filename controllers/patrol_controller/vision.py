"""
vision.py - The robot's AI "eyes".

1. grab a frame from the Webots camera (BGRA bytes -> numpy BGR image)
2. run YOLOv8 object detection on it (people, bags, suitcases)
3. camera + LiDAR fusion: turn a 2D bounding box into a position on the
   campus map (bearing from the box, distance from the LiDAR)
4. draw boxes and save annotated snapshots as evidence

For the Computer Vision subject this module is the one to upgrade
(tracking, a custom-trained model, evaluation metrics...).
"""
import math
import os

import numpy as np

try:
    import cv2
except ImportError:          # vision still loads; snapshots are just disabled
    cv2 = None

# COCO class ids used by YOLOv8 -> our label
CLASSES = {0: "person", 24: "backpack", 26: "handbag", 28: "suitcase"}
BAG_LABELS = {"backpack", "handbag", "suitcase"}
REAL_HEIGHT = {"person": 1.7, "backpack": 0.5, "handbag": 0.35, "suitcase": 0.65}  # metres
COLOURS = {"person": (0, 200, 255), "backpack": (255, 80, 200),
           "handbag": (255, 80, 200), "suitcase": (255, 80, 200)}     # BGR


class Detector:
    """Wrapper around an Ultralytics YOLO model."""

    def __init__(self, model_path, conf=0.30, bag_conf=0.20, imgsz=416):
        self.ok, self.error = False, None
        self.conf, self.bag_conf, self.imgsz = conf, bag_conf, imgsz
        self.class_map, self.custom = dict(CLASSES), False
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            names = {int(k): v for k, v in self.model.names.items()}
            if set(names.values()) <= {"person", "suitcase", "backpack", "handbag"}:
                self.class_map, self.custom = names, True       # our campus-trained model
            self.ok = True
        except Exception as e:                       # noqa: BLE001
            self.error = "%s: %s" % (type(e).__name__, e)

    def detect(self, frame):
        """Returns a list of dicts: label, conf, box (x1, y1, x2, y2) in pixels."""
        if not self.ok or frame is None:
            return []
        res = self.model.predict(frame, imgsz=self.imgsz, conf=min(self.conf, self.bag_conf),
                                 classes=list(self.class_map), verbose=False)[0]
        dets = []
        for box, conf, cls in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist(),
                                  res.boxes.cls.tolist()):
            label = self.class_map.get(int(cls))
            if label is None:
                continue
            need = self.bag_conf if label in BAG_LABELS else self.conf
            if conf >= need:
                dets.append({"label": label, "conf": round(float(conf), 2),
                             "box": [int(v) for v in box]})
        return dets


def camera_frame(camera):
    """Webots camera image (BGRA bytes) -> numpy BGR array (h, w, 3)."""
    raw = camera.getImage()
    if not raw:
        return None
    h, w = camera.getHeight(), camera.getWidth()
    return np.frombuffer(raw, np.uint8).reshape((h, w, 4))[:, :, :3].copy()


# ------------------------------------------------------------ low-light enhancement
def brightness(frame):
    """Mean brightness of an image, 0 (black) .. 255 (white)."""
    if cv2 is None or frame is None:
        return 255.0
    return float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())


_CLAHE = {}


def _clahe(clip, tile):
    key = (clip, tile)
    if key not in _CLAHE:                         # create once, reuse every frame
        _CLAHE[key] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    return _CLAHE[key]


def denoise(frame, method):
    """Remove camera sensor noise. Measured on our night frames (noise std on a flat
    dark patch; raw = 4.84):  median3 2.28 (0.6 ms) | median5 1.39 (1.2 ms) |
    bilateral 1.17 (8 ms, keeps edges sharp) | non-local means 4.95 (264 ms - no gain)."""
    if method == "median3":
        return cv2.medianBlur(frame, 3)
    if method == "median5":
        return cv2.medianBlur(frame, 5)
    if method == "bilateral":
        return cv2.bilateralFilter(frame, 7, 50, 7)
    return frame


def enhance(frame, denoise_method="bilateral", use_gamma=True, use_clahe=True,
            target=0.40, clahe_clip=2.0, tile=8):
    """Night-vision pipeline, applied before YOLO when the image is dark:

      1. denoise   - bilateral filter (default) averages each pixel with neighbours of
                     similar colour, so noise is smoothed but object edges stay sharp
      2. gamma     - out = in^gamma with gamma < 1 brightens dark pixels more than
                     bright ones; gamma is chosen automatically so the mean brightness
                     moves to `target` (0..1)
      3. CLAHE     - Contrast Limited Adaptive Histogram Equalisation on the L
                     (lightness) channel in LAB colour space: equalises the histogram
                     in small tiles so local detail appears; the clip limit stops it
                     from amplifying noise

    Each step can be switched off - that is how the ablation study is done.
    Returns (enhanced image, info dict)."""
    if cv2 is None or frame is None:
        return frame, {}
    before = brightness(frame)
    img = denoise(frame, denoise_method)
    gamma = 1.0
    if use_gamma:
        mean = max(1.0, before) / 255.0
        gamma = float(np.clip(np.log(target) / np.log(mean), 0.25, 1.0))
        lut = (np.power(np.arange(256) / 255.0, gamma) * 255).astype(np.uint8)
        img = cv2.LUT(img, lut)
    if use_clahe:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = _clahe(clahe_clip, tile).apply(l)
        img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    return img, {"brightness_before": round(before, 1), "brightness_after": round(brightness(img), 1),
                 "gamma": round(gamma, 2)}


# Pipelines compared in the ablation study (tools/enhancement_eval.py).
PIPELINES = {
    "raw":                   None,
    "gamma only":            dict(denoise_method=None, use_clahe=False),
    "CLAHE only":            dict(denoise_method=None, use_gamma=False),
    "median3+gamma+CLAHE":   dict(denoise_method="median3", target=0.45, clahe_clip=2.5),  # first version
    "median5+gamma+CLAHE":   dict(denoise_method="median5"),
    "bilateral+gamma+CLAHE": dict(denoise_method="bilateral"),                             # robot default
}


def run_pipeline(frame, name):
    cfg = PIPELINES[name]
    if cfg is None:
        return frame, {"brightness_before": round(brightness(frame), 1), "gamma": 1.0}
    return enhance(frame, **cfg)


def bearing_of(box, width, hfov):
    """Angle (rad, +ve = left) from the camera axis to the centre of a box."""
    cx = (box[0] + box[2]) / 2.0
    return math.atan((0.5 - cx / width) * 2.0 * math.tan(hfov / 2.0))


def distance_from_box(box, height, hfov, width, label):
    """Pinhole-camera distance estimate from the box height in pixels."""
    vfov = 2 * math.atan(math.tan(hfov / 2) * height / width)
    focal_px = (height / 2.0) / math.tan(vfov / 2.0)
    h_px = max(1.0, box[3] - box[1])
    return focal_px * REAL_HEIGHT.get(label, 1.0) / h_px


def lidar_range_at(ranges, lidar_fov, bearing, window=2):
    """Closest finite LiDAR range around a bearing (index 0 = leftmost ray)."""
    n = len(ranges)
    step = lidar_fov / max(1, n - 1)
    k = int(round((lidar_fov / 2 - bearing) / step))
    vals = [ranges[i] for i in range(max(0, k - window), min(n, k + window + 1))
            if not (math.isinf(ranges[i]) or math.isnan(ranges[i]))]
    return min(vals) if vals else None


def locate(det, robot_pose, cam, ranges, lidar_fov):
    """Sensor fusion: where on the campus is this detection?
    Uses the LiDAR distance when it agrees with the camera estimate
    (within 40 %), otherwise falls back to the camera-only estimate."""
    x, y, yaw = robot_pose
    b = bearing_of(det["box"], cam["width"], cam["hfov"])
    d_cam = distance_from_box(det["box"], cam["height"], cam["hfov"], cam["width"], det["label"])
    d_lidar = lidar_range_at(ranges, lidar_fov, b)
    if d_lidar is not None and abs(d_lidar - d_cam) <= 0.4 * d_cam:
        d, src = d_lidar, "lidar"
    else:
        d, src = d_cam, "camera"
    d += cam.get("offset", 0.0)                   # camera sits ahead of the robot centre
    gx, gy = x + d * math.cos(yaw + b), y + d * math.sin(yaw + b)
    return {"x": round(gx, 2), "y": round(gy, 2), "dist": round(d, 2),
            "bearing": round(b, 3), "range_source": src}


def annotate(frame, dets, header, footer=None):
    """Draw boxes + labels and a header bar. Returns a new image."""
    if cv2 is None or frame is None:
        return None
    img = frame.copy()
    for d in dets:
        x1, y1, x2, y2 = d["box"]
        col = COLOURS.get(d["label"], (0, 255, 0))
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
        text = ("#%d " % d["track_id"] if d.get("track_id") else "") + "%s %.2f" % (d["label"], d["conf"])
        if "pos" in d:
            text += " (%.1fm)" % d["pos"]["dist"]
        cv2.putText(img, text, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)
    cv2.rectangle(img, (0, 0), (img.shape[1], 18), (0, 0, 0), -1)
    cv2.putText(img, header, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    if footer:
        h = img.shape[0]
        cv2.rectangle(img, (0, h - 18), (img.shape[1], h), (0, 0, 160), -1)
        cv2.putText(img, footer, (4, h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def save_image(img, path):
    if cv2 is None or img is None:
        return False
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp.jpg"
        cv2.imwrite(tmp, img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        os.replace(tmp, path)
        return True
    except (OSError, cv2.error):
        return False
