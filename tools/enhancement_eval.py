"""
enhancement_eval.py - Ablation study: which low-light enhancement helps YOLO at night?

Takes the raw night frames saved by the robot (output/run_.../frames_raw) and runs
YOLOv8 on every frame after each enhancement pipeline in vision.PIPELINES:

    raw | gamma only | CLAHE only | median3+gamma+CLAHE | median5+gamma+CLAHE | bilateral+gamma+CLAHE

Turning steps on and off one at a time ("ablation") shows which step helps and
which hurts (e.g. gamma brightens the image but also amplifies sensor noise).

Run from the VisionSecurityRobot folder (VS Code terminal):
    python tools/enhancement_eval.py                      (uses the latest run)
    python tools/enhancement_eval.py output/run_XXXX      (a specific run)

Writes to  <run>/enhancement_eval/ :
    results.csv          one row per frame x pipeline
    summary.txt          the comparison table for the report
    pipelines_grid.jpg   one frame through every pipeline, with YOLO boxes
    comparison_grid.jpg  raw vs the robot's default pipeline on 4 frames
    histograms.png       brightness histograms raw vs default pipeline
    noise_vs_brightness.png  how much each pipeline brightens vs amplifies noise
"""
import csv
import glob
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "controllers", "patrol_controller"))
import vision as V  # noqa: E402  (exactly the code the robot runs)

CONF = 0.25
CLASSES = {0: "person", 24: "backpack", 26: "handbag", 28: "suitcase"}
DEFAULT = "bilateral+gamma+CLAHE"


def latest_run():
    runs = sorted(glob.glob(os.path.join(ROOT, "output", "run_*")))
    runs = [r for r in runs if glob.glob(os.path.join(r, "frames_raw", "*.jpg"))]
    if not runs:
        sys.exit("No run with saved frames found. Run the Webots simulation first.")
    return runs[-1]


def detect(model, img):
    if model is None:
        return []
    r = model.predict(img, imgsz=416, conf=CONF, classes=list(CLASSES), verbose=False)[0]
    return [{"label": CLASSES[int(c)], "conf": float(p), "box": [int(v) for v in b]}
            for b, p, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())]


def noise_level(img):
    """Sensor-noise estimate: median absolute deviation of the image minus a blurred
    copy (i.e. of the fine, high-frequency detail), on the grey image."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hf = g - cv2.GaussianBlur(g, (5, 5), 0)
    return float(np.median(np.abs(hf - np.median(hf))) * 1.4826)


def draw(img, dets, title):
    out = img.copy()
    for d in dets:
        x1, y1, x2, y2 = d["box"]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 255), 2)
        cv2.putText(out, "%s %.2f" % (d["label"], d["conf"]), (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)
    cv2.rectangle(out, (0, 0), (out.shape[1], 20), (0, 0, 0), -1)
    cv2.putText(out, title, (5, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else latest_run()
    frames = sorted(glob.glob(os.path.join(run, "frames_raw", "*.jpg")))
    if not frames:
        sys.exit("No frames in %s/frames_raw" % run)
    out_dir = os.path.join(run, "enhancement_eval")
    os.makedirs(out_dir, exist_ok=True)

    try:
        from ultralytics import YOLO
        mp = os.path.join(ROOT, "models", "yolov8n.pt")
        model = YOLO(mp if os.path.exists(mp) else "yolov8n.pt")
    except Exception as e:                                     # noqa: BLE001
        print("YOLO not available (%s) - only image statistics will be computed" % e)
        model = None

    names = list(V.PIPELINES)
    for n in names:                                             # warm-up (CLAHE objects etc.)
        V.run_pipeline(cv2.imread(frames[0]), n)
    print("Ablation study: %d frames x %d pipelines from %s" % (len(frames), len(names), run))

    rows, per_frame = [], []
    for i, path in enumerate(frames):
        raw = cv2.imread(path)
        res = {}
        for n in names:
            t0 = time.perf_counter()
            img, info = V.run_pipeline(raw, n)
            ms = (time.perf_counter() - t0) * 1000
            dets = detect(model, img)
            res[n] = (img, dets)
            rows.append({"frame": os.path.basename(path), "pipeline": n,
                         "brightness": round(V.brightness(img), 1), "noise": round(noise_level(img), 2),
                         "ms": round(ms, 2), "detections": len(dets),
                         "mean_conf": round(float(np.mean([d["conf"] for d in dets])), 3) if dets else 0.0})
        per_frame.append((os.path.basename(path), res))
        if (i + 1) % 20 == 0:
            print("  %d / %d" % (i + 1, len(frames)))

    with open(os.path.join(out_dir, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # ------------------------------------------------------------- summary table
    nf = len(frames)
    stats = {}
    for n in names:
        r = [x for x in rows if x["pipeline"] == n]
        confs = [x["mean_conf"] for x in r if x["detections"]]
        stats[n] = {"bright": np.mean([x["brightness"] for x in r]), "noise": np.mean([x["noise"] for x in r]),
                    "ms": np.median([x["ms"] for x in r]),
                    "frames_det": 100.0 * sum(1 for x in r if x["detections"]) / nf,
                    "dets": sum(x["detections"] for x in r), "conf": np.mean(confs) if confs else 0.0}
    hdr = "%-22s %8s %7s %9s %9s %6s %7s" % ("pipeline", "bright", "noise", "time(ms)", "frames%", "dets", "conf")
    lines = ["LOW-LIGHT ENHANCEMENT - ABLATION STUDY", "run: %s   frames: %d" % (os.path.basename(run), nf), "",
             hdr, "-" * len(hdr)]
    for n in names:
        s = stats[n]
        lines.append("%-22s %8.1f %7.2f %9.1f %8.1f%% %6d %7.3f%s" % (
            n, s["bright"], s["noise"], s["ms"], s["frames_det"], s["dets"], s["conf"],
            "   <- robot default" if n == DEFAULT else ""))
    lines += ["",
              "bright   = mean brightness 0-255        noise = sensor-noise estimate (lower = cleaner)",
              "frames%  = frames with >= 1 detection   dets  = total detections   conf = mean confidence",
              "",
              "Note: counts without ground-truth labels cannot separate true from false detections.",
              "Precision / recall / mAP on labelled frames are computed in the custom-model step."]
    summary = "\n".join(lines)
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(summary + "\n")
    print("\n" + summary)

    # ------------------------------------------------------------- images
    def score(item):                     # frame where pipelines disagree most is the most instructive
        counts = [len(item[1][n][1]) for n in names]
        return (max(counts) - min(counts), sum(counts))
    best = sorted(per_frame, key=score, reverse=True)
    name0, res0 = best[0]
    tiles = [draw(res0[n][0], res0[n][1], "%s (%d det)" % (n, len(res0[n][1]))) for n in names]
    cv2.imwrite(os.path.join(out_dir, "pipelines_grid.jpg"),
                np.vstack([np.hstack(tiles[0:3]), np.hstack(tiles[3:6])]))
    rows_img = []
    for fname, res in best[:4]:
        rows_img.append(np.hstack([draw(res["raw"][0], res["raw"][1], "RAW %s (%d det)" % (fname[:18], len(res["raw"][1]))),
                                   draw(res[DEFAULT][0], res[DEFAULT][1], "%s (%d det)" % (DEFAULT, len(res[DEFAULT][1])))]))
    cv2.imwrite(os.path.join(out_dir, "comparison_grid.jpg"), np.vstack(rows_img))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
        for n, col in (("raw", "#888888"), (DEFAULT, "#2a7fd4")):
            g = cv2.cvtColor(res0[n][0], cv2.COLOR_BGR2GRAY)
            ax[0].hist(g.ravel(), bins=64, range=(0, 255), alpha=0.65, color=col, label=n)
            ax[1].hist([x["brightness"] for x in rows if x["pipeline"] == n], bins=20, alpha=0.65, color=col, label=n)
        ax[0].set_title("Pixel histogram of one frame"); ax[0].set_xlabel("brightness (0-255)"); ax[0].legend()
        ax[1].set_title("Mean brightness of all %d frames" % nf); ax[1].set_xlabel("mean brightness"); ax[1].legend()
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, "histograms.png"), dpi=130); plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for n in names:
            ax.scatter(stats[n]["bright"], stats[n]["noise"], s=80)
            ax.annotate(n, (stats[n]["bright"], stats[n]["noise"]), textcoords="offset points", xytext=(6, 4), fontsize=8)
        ax.set_xlabel("mean brightness (higher = brighter)"); ax.set_ylabel("noise (lower = cleaner)")
        ax.set_title("Brightness gained vs noise amplified"); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, "noise_vs_brightness.png"), dpi=130); plt.close(fig)
    except ImportError:
        print("(matplotlib not installed - charts skipped)")
    print("\nSaved results to", out_dir)


if __name__ == "__main__":
    main()
