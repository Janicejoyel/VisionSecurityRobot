"""
dataset_collector.py - Generates a labelled object-detection dataset automatically.

Runs on the "datacam" robot in worlds/dataset_capture.wbt. It is a SUPERVISOR, so it
can move itself, move the suitcases, change the lighting and read the exact position
of every pedestrian. For each sample it:

  1. picks a lighting condition (day / dusk / night) and sets sky + sunlight
  2. scatters the 6 suitcases over free ground (random place, rotation)
  3. chooses a target (a person, a suitcase, or nothing) and places the camera
     2-14 m away, looking roughly at it (random offset so objects are not always centred)
  4. renders the camera image + depth image
  5. auto-labels every person and suitcase (autolabel.py: projection + depth + occlusion)
  6. adds realistic sensor noise (stronger at night) and saves image + YOLO label

Output: <project>/dataset/{train,val,test_night}/{images,labels}, data.yaml, meta.jsonl,
previews/ (a few images with the labels drawn, to check them by eye).
"""
import json
import math
import os
import random
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "controllers", "patrol_controller"))
import campus_layout as L  # noqa: E402
from costmap import CostMap  # noqa: E402
import autolabel as A  # noqa: E402

from controller import Supervisor  # noqa: E402

try:
    import cv2
except ImportError:
    cv2 = None

CLASSES = ["person", "suitcase"]
OUT = os.path.join(ROOT, "dataset")
STEPS_PER_SAMPLE = 8          # simulation steps between samples (people keep walking)
PREVIEW_EVERY = 40

LIGHTING = {   # name: (sun intensity, ambient, sky colour, sky luminosity, noise std range)
    "day":   ((1.6, 2.3), (0.45, 0.65), (0.55, 0.72, 0.90), 1.0, (0.0, 0.008)),
    "dusk":  ((0.45, 0.8), (0.18, 0.3), (0.35, 0.30, 0.40), 0.5, (0.01, 0.025)),
    "night": ((0.08, 0.2), (0.04, 0.10), (0.01, 0.015, 0.04), 0.15, (0.025, 0.05)),
}


def arg(key, default):
    for a in sys.argv[1:]:
        if a.startswith("--%s=" % key):
            return int(a.split("=", 1)[1])
    return default


class Collector:
    def __init__(self):
        self.sv = Supervisor()
        self.dt = int(self.sv.getBasicTimeStep())
        self.cam = self.sv.getDevice("camera")
        self.cam.enable(self.dt)
        self.rf = self.sv.getDevice("depth")
        self.rf.enable(self.dt)
        self.W, self.H = self.cam.getWidth(), self.cam.getHeight()
        self.fov = self.cam.getFov()
        self.f = A.focal_px(self.W, self.fov)
        self.me = self.sv.getSelf()
        self.f_trans, self.f_rot = self.me.getField("translation"), self.me.getField("rotation")
        self.sun = self.sv.getFromDef("SUN")
        self.sky = self.sv.getFromDef("SKY")
        self.peds = [(n, self.sv.getFromDef(n.upper())) for n, *_ in L.DATASET_PEDESTRIANS]
        self.suitcases = [(self.sv.getFromDef("SUITCASE_%d" % (k + 1)), size)
                          for k, (size, _) in enumerate(L.DATASET_SUITCASES)]
        self.map = CostMap(inflation=0.5)
        self.free = np.argwhere(~self.map.blocked)
        self.depth_kind = "planar"
        self.suitcase_poses = []

    # ------------------------------------------------------------ helpers
    def step(self, n=1):
        for _ in range(n):
            if self.sv.step(self.dt) == -1:
                sys.exit(0)

    def random_free(self, near=None, rmin=0.0, rmax=99.0, tries=60):
        for _ in range(tries):
            if near is None:
                i, j = self.free[random.randrange(len(self.free))]
                x, y = self.map.cell_to_world(i, j)
                x += random.uniform(-0.15, 0.15); y += random.uniform(-0.15, 0.15)
            else:
                a, r = random.uniform(0, 2 * math.pi), random.uniform(rmin, rmax)
                x, y = near[0] + r * math.cos(a), near[1] + r * math.sin(a)
            if self.map.in_grid(x, y) and not self.map.is_blocked(*self.map.world_to_cell(x, y)):
                return x, y
        return None

    def set_camera(self, x, y, z, yaw):
        self.f_trans.setSFVec3f([x, y, z])
        self.f_rot.setSFRotation([0, 0, 1, yaw])

    def set_lighting(self, name):
        si, amb, sky, lum, noise = LIGHTING[name]
        self.sun.getField("intensity").setSFFloat(random.uniform(*si))
        self.sun.getField("ambientIntensity").setSFFloat(random.uniform(*amb))
        self.sky.getField("skyColor").setMFColor(0, list(sky))
        self.sky.getField("luminosity").setSFFloat(lum)
        return random.uniform(*noise)

    def scatter_suitcases(self):
        self.suitcase_poses = []
        for node, size in self.suitcases:
            if random.random() < 0.75:
                p = self.random_free()
            else:
                p = None
            if p is None:                                   # park it off-campus
                p = (25.0 + random.uniform(0, 5), 25.0)
            yaw = random.uniform(-math.pi, math.pi)
            node.getField("translation").setSFVec3f([p[0], p[1], size[2] / 2 + 0.06])
            node.getField("rotation").setSFRotation([0, 0, 1, yaw])
            node.resetPhysics()
            self.suitcase_poses.append((p[0], p[1], yaw, size))

    def objects(self):
        objs = []
        for name, node in self.peds:
            x, y, _ = node.getPosition()
            objs.append(dict(cls=0, name=name, x=x, y=y, yaw=0.0, sx=0.7, sy=0.7, z0=0.0, z1=1.85, fill=0.35))
        for k, (x, y, yaw, (sx, sy, sz)) in enumerate(self.suitcase_poses):
            objs.append(dict(cls=1, name="suitcase_%d" % (k + 1), x=x, y=y, yaw=yaw, sx=sx, sy=sy,
                             z0=0.02, z1=sz + 0.12, fill=0.7))
        return objs

    def grab(self):
        raw = self.cam.getImage()
        img = np.frombuffer(raw, np.uint8).reshape((self.H, self.W, 4))[:, :, :3].copy()
        depth = None
        data = self.rf.getRangeImage()                          # standard: flat list, row by row
        if data is not None and len(data) == self.W * self.H:
            depth = np.asarray(data, np.float64).reshape((self.H, self.W))
        else:                                                   # backup: 2-D array [x][y]
            arr = np.asarray(self.rf.getRangeImageArray(), np.float64)
            depth = arr.T if arr.shape == (self.W, self.H) else arr.reshape((self.H, self.W))
        depth[depth >= self.rf.getMaxRange() * 0.999] = np.inf
        return img, depth

    def calibrate_depth(self):
        """Does the RangeFinder return planar depth or distance along the ray?
        Look at flat ground from a known height and compare with both models."""
        self.set_lighting("day")
        self.set_camera(0.0, -14.0, 0.5, math.pi / 2)        # main avenue, looking north
        self.step(3)
        _, depth = self.grab()
        v = self.H - 12
        u = np.array([20, self.W - 21])                        # left and right edges
        measured = depth[v, u]
        planar = 0.5 * self.f / (v - self.H / 2.0)
        ray = planar * np.sqrt(1 + ((u - self.W / 2.0) / self.f) ** 2 + ((v - self.H / 2.0) / self.f) ** 2)
        e_planar = float(np.nanmean(np.abs(measured - planar)))
        e_ray = float(np.nanmean(np.abs(measured - ray)))
        self.depth_kind = "planar" if e_planar <= e_ray else "ray"
        print("[datacam] depth calibration: measured %s, planar %.3f (err %.3f), ray %s (err %.3f) -> %s"
              % (np.round(measured, 3), planar, e_planar, np.round(ray, 3), e_ray, self.depth_kind))

    # ------------------------------------------------------------ one sample
    def sample(self, lighting):
        noise = self.set_lighting(lighting)
        self.scatter_suitcases()
        self.step(STEPS_PER_SAMPLE)                            # people walk on a bit
        objs = self.objects()
        r = random.random()
        if r < 0.45:
            target = random.choice([o for o in objs if o["cls"] == 0])
            dist = (2.5, 14.0)
        elif r < 0.85:
            placed = [o for o in objs if o["cls"] == 1 and abs(o["x"]) < 21]
            target = random.choice(placed) if placed else None
            dist = (1.5, 8.0)
        else:
            target = None
        if target is not None:
            p = self.random_free(near=(target["x"], target["y"]), rmin=dist[0], rmax=dist[1])
        else:
            p = None
        if p is None:
            p = self.random_free()
            yaw = random.uniform(-math.pi, math.pi)
        else:
            yaw = math.atan2(target["y"] - p[1], target["x"] - p[0]) + random.uniform(-0.45, 0.45)
        z = random.uniform(0.42, 0.62)
        self.set_camera(p[0], p[1], z, yaw)
        self.step(2)                                           # render from the new pose
        objs = self.objects()                                  # positions at render time
        img, depth = self.grab()
        dp = A.planar_depth(depth, self.depth_kind, self.W, self.H, self.f)
        cam = (p[0], p[1], z, yaw)
        labels = [lab for o in objs for lab in [A.label_object(o, cam, dp, self.W, self.H, self.fov)] if lab]
        if noise > 0:
            img = np.clip(img.astype(np.float32) + np.random.normal(0, noise * 255, img.shape), 0, 255).astype(np.uint8)
        return img, labels, {"lighting": lighting, "noise": round(noise, 4), "cam": [round(c, 3) for c in cam],
                             "brightness": round(float(img.mean()), 1)}

    # ------------------------------------------------------------ main
    def run(self):
        splits = [("train", arg("train", 600), None), ("val", arg("val", 150), None),
                  ("test_night", arg("test_night", 150), "night")]
        if os.path.isdir(OUT):
            shutil.rmtree(OUT, ignore_errors=True)
        for s, _, _ in splits:
            os.makedirs(os.path.join(OUT, s, "images"), exist_ok=True)
            os.makedirs(os.path.join(OUT, s, "labels"), exist_ok=True)
        os.makedirs(os.path.join(OUT, "previews"), exist_ok=True)
        random.seed(7); np.random.seed(7)
        self.step(2)
        self.calibrate_depth()
        meta = open(os.path.join(OUT, "meta.jsonl"), "w")
        counts = {s: [0, 0, 0] for s, _, _ in splits}           # images, persons, suitcases
        t0, k = time.time(), 0
        total = sum(n for _, n, _ in splits)
        for split, n, forced in splits:
            for i in range(n):
                lighting = forced or random.choices(["day", "dusk", "night"], [0.35, 0.3, 0.35])[0]
                img, labels, info = self.sample(lighting)
                name = "%s_%05d" % (split, i)
                cv2.imwrite(os.path.join(OUT, split, "images", name + ".jpg"), img,      # img is BGR
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
                with open(os.path.join(OUT, split, "labels", name + ".txt"), "w") as f:
                    f.write("\n".join(A.to_yolo(l, self.W, self.H) for l in labels) + ("\n" if labels else ""))
                meta.write(json.dumps(dict(info, image=name, split=split, labels=labels)) + "\n")
                counts[split][0] += 1
                counts[split][1] += sum(1 for l in labels if l["cls"] == 0)
                counts[split][2] += sum(1 for l in labels if l["cls"] == 1)
                k += 1
                if k % PREVIEW_EVERY == 1:
                    self.save_preview(img, labels, name, info)
                if k % 25 == 0:
                    el = time.time() - t0
                    print("[datacam] %d / %d samples (%.0f s elapsed, ~%.0f s left)" % (k, total, el, el / k * (total - k)))
        meta.close()
        with open(os.path.join(OUT, "data.yaml"), "w") as f:
            f.write("# InteliPatrol / Vision Security Robot - auto-labelled campus dataset\n")
            f.write("path: %s\ntrain: train/images\nval: val/images\ntest: test_night/images\n" % OUT.replace("\\", "/"))
            f.write("names:\n  0: person\n  1: suitcase\n")
        print("[datacam] DONE. images / person boxes / suitcase boxes per split:")
        for s, (ni, npers, nsc) in counts.items():
            print("[datacam]   %-10s %4d images  %5d person  %5d suitcase" % (s, ni, npers, nsc))
        print("[datacam] dataset saved in", OUT)
        self.sv.simulationSetMode(Supervisor.SIMULATION_MODE_PAUSE)

    def save_preview(self, img, labels, name, info):
        if cv2 is None:
            return
        out = img.copy()                                       # already BGR (OpenCV order)
        for l in labels:
            x0, y0, x1, y1 = [int(v) for v in l["box"]]
            col = (0, 220, 255) if l["cls"] == 0 else (255, 80, 200)
            cv2.rectangle(out, (x0, y0), (x1, y1), col, 2)
            cv2.putText(out, "%s %.1fm" % (CLASSES[l["cls"]], l["dist"]), (x0, max(12, y0 - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
        cv2.putText(out, "%s  %s  noise %.3f" % (name, info["lighting"], info["noise"]), (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(OUT, "previews", name + ".jpg"), out)


if __name__ == "__main__":
    Collector().run()
