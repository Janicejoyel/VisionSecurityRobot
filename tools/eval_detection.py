"""
eval_detection.py - Precision / recall / mAP of a detector on the labelled dataset.

How a prediction is scored
  * A prediction is a TRUE POSITIVE (TP) if it overlaps an unmatched ground-truth box
    of the same class with IoU >= 0.5   (IoU = area of overlap / area of union).
  * Otherwise it is a FALSE POSITIVE (FP). Ground-truth boxes nobody matched are
    FALSE NEGATIVES (FN, i.e. missed objects).
  * precision = TP / (TP + FP)   "of what it detected, how much was right"
    recall    = TP / (TP + FN)   "of what was there, how much it found"
  * AP@0.5   = area under the precision-recall curve (all confidence thresholds)
    mAP@0.5  = mean AP over the classes

Usage (VisionSecurityRobot folder, VS Code terminal):
    python tools/eval_detection.py                                   pretrained YOLOv8n, test_night, all pipelines
    python tools/eval_detection.py --model models/campus_yolo.pt     our custom model
    python tools/eval_detection.py --split val --pipelines raw       other split / one pipeline

The pretrained model knows COCO classes; for a fair comparison its 'suitcase',
'backpack' and 'handbag' predictions all count as our 'suitcase' class.
Results are written to results/<model>_<split>/.
"""
import argparse
import csv
import glob
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "controllers", "patrol_controller"))
import vision as V  # noqa: E402

CLASSES = ["person", "suitcase"]
COCO_MAP = {0: 0, 28: 1, 24: 1, 26: 1}          # COCO id -> our class id
DIST_BINS = [(0, 6, "near (<6 m)"), (6, 12, "mid (6-12 m)"), (12, 99, "far (>12 m)")]


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def load_ground_truth(split_dir, meta_path):
    """Ground truth from meta.jsonl (has distances), falling back to YOLO .txt labels."""
    gt = {}
    if os.path.exists(meta_path):
        for line in open(meta_path):
            m = json.loads(line)
            if m["split"] == os.path.basename(split_dir):
                gt[m["image"]] = [{"cls": l["cls"], "box": l["box"], "dist": l["dist"]} for l in m["labels"]]
    if gt:
        return gt
    for img in sorted(glob.glob(os.path.join(split_dir, "images", "*.jpg"))):
        name = os.path.splitext(os.path.basename(img))[0]
        h, w = cv2.imread(img).shape[:2]
        boxes = []
        lab = os.path.join(split_dir, "labels", name + ".txt")
        if os.path.exists(lab):
            for row in open(lab):
                c, cx, cy, bw, bh = map(float, row.split())
                boxes.append({"cls": int(c), "box": [(cx - bw / 2) * w, (cy - bh / 2) * h,
                                                      (cx + bw / 2) * w, (cy + bh / 2) * h], "dist": -1})
        gt[name] = boxes
    return gt


def predict(model, img, coco):
    r = model.predict(img, imgsz=416, conf=0.01, verbose=False)[0]
    out = []
    for b, p, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
        c = int(c)
        if coco:
            if c not in COCO_MAP:
                continue
            c = COCO_MAP[c]
        if c < len(CLASSES):
            out.append({"cls": c, "box": b, "conf": float(p)})
    return out


def evaluate(gt, preds, conf_thr=0.25):
    """Returns per-class stats + recall by distance."""
    res = {}
    for c, cname in enumerate(CLASSES):
        n_gt = sum(1 for boxes in gt.values() for g in boxes if g["cls"] == c)
        scored = []                                  # (conf, is_tp) for the PR curve
        tp = fp = 0
        found_by_dist = {b[2]: [0, 0] for b in DIST_BINS}
        for name, boxes in gt.items():
            g = [x for x in boxes if x["cls"] == c]
            used = [False] * len(g)
            for p in sorted([x for x in preds.get(name, []) if x["cls"] == c], key=lambda x: -x["conf"]):
                best, bi = 0.0, -1
                for k, gg in enumerate(g):
                    if not used[k]:
                        o = iou(p["box"], gg["box"])
                        if o > best:
                            best, bi = o, k
                hit = best >= 0.5
                if hit:
                    used[bi] = True
                scored.append((p["conf"], hit))
                if p["conf"] >= conf_thr:
                    if hit:
                        tp += 1
                        gg = g[bi]
                        for lo, hi, lab in DIST_BINS:
                            if lo <= gg["dist"] < hi:
                                found_by_dist[lab][0] += 1
                    else:
                        fp += 1
            for gg in g:
                for lo, hi, lab in DIST_BINS:
                    if lo <= gg["dist"] < hi:
                        found_by_dist[lab][1] += 1
        scored.sort(key=lambda x: -x[0])
        tps = np.cumsum([s[1] for s in scored]) if scored else np.array([0])
        fps = np.cumsum([not s[1] for s in scored]) if scored else np.array([0])
        rec = tps / max(1, n_gt)
        prec = tps / np.maximum(1, tps + fps)
        mrec = np.concatenate([[0], rec, [1]])
        mpre = np.concatenate([[1], prec, [0]])
        for i in range(len(mpre) - 2, -1, -1):        # precision envelope
            mpre[i] = max(mpre[i], mpre[i + 1])
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])) if n_gt else float("nan")
        res[cname] = {"gt": n_gt, "tp": tp, "fp": fp, "fn": n_gt - tp,
                      "precision": tp / max(1, tp + fp), "recall": tp / max(1, n_gt), "ap50": ap,
                      "pr": (rec.tolist(), prec.tolist()),
                      "recall_by_dist": {k: (v[0] / v[1] if v[1] else float("nan"), v[1])
                                         for k, v in found_by_dist.items()}}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "models", "yolov8n.pt"))
    ap.add_argument("--split", default="test_night")
    ap.add_argument("--pipelines", default="all", help="'all' or comma list of names from vision.PIPELINES")
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    split_dir = os.path.join(ROOT, "dataset", args.split)
    if not os.path.isdir(split_dir):
        sys.exit("No dataset at %s - run worlds/dataset_capture.wbt first." % split_dir)
    gt = load_ground_truth(split_dir, os.path.join(ROOT, "dataset", "meta.jsonl"))
    names = sorted(gt)
    pipes = list(V.PIPELINES) if args.pipelines == "all" else args.pipelines.split(",")

    from ultralytics import YOLO
    model = YOLO(args.model)
    coco = len(model.names) > 2                      # pretrained COCO model vs our 2-class model
    tag = os.path.splitext(os.path.basename(args.model))[0]
    out_dir = os.path.join(ROOT, "results", "%s_%s" % (tag, args.split))
    os.makedirs(out_dir, exist_ok=True)
    print("Evaluating %s on %s (%d images, %d person / %d suitcase boxes), pipelines: %s"
          % (tag, args.split, len(names), sum(1 for b in gt.values() for g in b if g["cls"] == 0),
             sum(1 for b in gt.values() for g in b if g["cls"] == 1), ", ".join(pipes)))

    all_res = {}
    for pipe in pipes:
        preds = {}
        for k, name in enumerate(names):
            img = cv2.imread(os.path.join(split_dir, "images", name + ".jpg"))
            img, _ = V.run_pipeline(img, pipe)
            preds[name] = predict(model, img, coco)
        all_res[pipe] = evaluate(gt, preds, args.conf)
        r = all_res[pipe]
        print("  %-22s person P %.2f R %.2f AP50 %.3f | suitcase P %.2f R %.2f AP50 %.3f" % (
            pipe, r["person"]["precision"], r["person"]["recall"], r["person"]["ap50"],
            r["suitcase"]["precision"], r["suitcase"]["recall"], r["suitcase"]["ap50"]))

    # ------------------------------------------------------------- summary
    hdr = "%-22s | %-27s | %-27s | %7s" % ("pipeline", "person  P / R / AP50", "suitcase  P / R / AP50", "mAP50")
    lines = ["DETECTION EVALUATION (IoU >= 0.5, confidence >= %.2f)" % args.conf,
             "model: %s   split: %s   images: %d" % (tag, args.split, len(names)), "", hdr, "-" * len(hdr)]
    rows = []
    for pipe, r in all_res.items():
        aps = [r[c]["ap50"] for c in CLASSES if r[c]["gt"]]
        m = float(np.mean(aps)) if aps else float("nan")
        lines.append("%-22s | %.2f / %.2f / %.3f        | %.2f / %.2f / %.3f        | %.3f" % (
            pipe, r["person"]["precision"], r["person"]["recall"], r["person"]["ap50"],
            r["suitcase"]["precision"], r["suitcase"]["recall"], r["suitcase"]["ap50"], m))
        for c in CLASSES:
            rows.append({"pipeline": pipe, "class": c, **{k: r[c][k] for k in ("gt", "tp", "fp", "fn")},
                         "precision": round(r[c]["precision"], 4), "recall": round(r[c]["recall"], 4),
                         "ap50": round(r[c]["ap50"], 4), "map50": round(m, 4)})
    best = max(all_res, key=lambda p: np.nanmean([all_res[p][c]["ap50"] for c in CLASSES]))
    lines += ["", "Person recall by distance (pipeline: %s):" % best]
    for lab, (rv, n) in all_res[best]["person"]["recall_by_dist"].items():
        lines.append("  %-14s %5.1f%%  (%d people)" % (lab, 100 * rv if rv == rv else 0, n))
    lines += ["", "best pipeline by mAP50: %s" % best]
    summary = "\n".join(lines)
    open(os.path.join(out_dir, "summary.txt"), "w").write(summary + "\n")
    with open(os.path.join(out_dir, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print("\n" + summary)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
        for ax, c in zip(axes, CLASSES):
            for pipe, r in all_res.items():
                rec, prec = r[c]["pr"]
                if len(rec) > 1:
                    ax.plot(rec, prec, label="%s (AP %.2f)" % (pipe, r[c]["ap50"]))
            ax.set_title("Precision-recall: %s" % c); ax.set_xlabel("recall"); ax.set_ylabel("precision")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.grid(alpha=.3); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, "pr_curves.png"), dpi=130)
    except ImportError:
        pass
    print("\nSaved to", out_dir)


if __name__ == "__main__":
    main()
