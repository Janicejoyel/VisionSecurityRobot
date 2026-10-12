"""
make_report_charts.py - Final comparison figures for the report.

Reads the evaluation results of both models (run tools/eval_detection.py for each first):
    results/yolov8n_test_night/        COCO-pretrained YOLOv8n
    results/campus_yolo_test_night/    our campus-trained model
and writes results/report_charts/:
    fig1_map_by_pipeline.png     mAP@0.5 of both models under every enhancement pipeline
    fig2_per_class.png           AP@0.5 and recall per class (raw images)
    fig3_recall_by_distance.png  person recall: near / mid / far
    summary_table.txt            the numbers behind the figures

Run from the VisionSecurityRobot folder:  python tools/make_report_charts.py
"""
import csv
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "report_charts")
MODELS = [("yolov8n_test_night", "Pretrained YOLOv8n (COCO)", "#eb6834"),
          ("campus_yolo_test_night", "Campus-trained YOLOv8n (ours)", "#2a78d6")]
INK, INK2, GRID = "#1f1f1e", "#5f5e5a", "#e4e2dc"
PIPE_SHORT = {"raw": "raw", "gamma only": "gamma", "CLAHE only": "CLAHE",
              "median3+gamma+CLAHE": "median3\n+γ+CLAHE", "median5+gamma+CLAHE": "median5\n+γ+CLAHE",
              "bilateral+gamma+CLAHE": "bilateral\n+γ+CLAHE"}


def load(tag):
    path = os.path.join(RES, tag, "results.csv")
    if not os.path.exists(path):
        sys.exit("Missing %s - run tools/eval_detection.py for this model first." % path)
    rows = list(csv.DictReader(open(path)))
    dist = {}
    summ = os.path.join(RES, tag, "summary.txt")
    if os.path.exists(summ):
        txt = open(summ).read()
        pipe = re.search(r"Person recall by distance \(pipeline: (.+?)\)", txt)
        for lab, val in re.findall(r"^\s+(near|mid|far) \([^)]*\)\s+([\d.]+)%", txt, re.M):
            dist[lab] = float(val) / 100
        dist["_pipeline"] = pipe.group(1) if pipe else "?"
    return rows, dist


def style(ax, title, ylabel):
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=12, fontweight="bold")
    ax.set_ylabel(ylabel, color=INK2)
    ax.set_ylim(0, 1.22)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#bdbbb4")
    ax.tick_params(colors=INK2, length=0)


def bars(ax, groups, series, labels_at=None, width=0.38, gap=0.02):
    """series: list of (name, colour, values). Thin gap between neighbouring bars."""
    n = len(series)
    for k, (name, col, vals) in enumerate(series):
        xs = [i + (k - (n - 1) / 2) * (width + gap) for i in range(len(groups))]
        b = ax.bar(xs, vals, width=width, color=col, label=name, zorder=3)
        for i, (x, v) in enumerate(zip(xs, vals)):
            if labels_at is None or i in labels_at:
                ax.text(x, v + 0.015, "%.2f" % v, ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, color=INK2, fontsize=9)
    ax.legend(frameon=False, loc="upper left", ncol=2, fontsize=8.5, labelcolor=INK,
              handlelength=1.2, columnspacing=1.2)


def main():
    os.makedirs(OUT, exist_ok=True)
    data = {tag: load(tag) for tag, _, _ in MODELS}
    pipes = list(dict.fromkeys(r["pipeline"] for r in data[MODELS[0][0]][0]))

    def val(tag, pipe, cls, key):
        for r in data[tag][0]:
            if r["pipeline"] == pipe and r["class"] == cls:
                return float(r[key])
        return float("nan")

    # ---- fig 1: mAP50 by pipeline
    fig, ax = plt.subplots(figsize=(10, 4.6))
    series = [(name, col, [val(tag, p, "person", "map50") for p in pipes]) for tag, name, col in MODELS]
    bars(ax, [PIPE_SHORT.get(p, p) for p in pipes], series)
    style(ax, "mAP@0.5 on the night test set, by enhancement pipeline", "mAP@0.5")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig1_map_by_pipeline.png"), dpi=160); plt.close(fig)

    # ---- fig 2: per class AP50 and recall on raw images
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, key, title in zip(axes, ("ap50", "recall"), ("AP@0.5 per class (raw images)", "Recall per class (raw images)")):
        series = [(name, col, [val(tag, "raw", c, key) for c in ("person", "suitcase")]) for tag, name, col in MODELS]
        bars(ax, ["person", "suitcase"], series)
        style(ax, title, key.upper() if key == "ap50" else "recall")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig2_per_class.png"), dpi=160); plt.close(fig)

    # ---- fig 3: person recall by distance
    fig, ax = plt.subplots(figsize=(8, 4.2))
    groups = ["near (<6 m)", "mid (6-12 m)", "far (>12 m)"]
    series = [("%s - %s" % (name, data[tag][1].get("_pipeline", "")), col,
               [data[tag][1].get(k, float("nan")) for k in ("near", "mid", "far")]) for tag, name, col in MODELS]
    bars(ax, groups, series)
    style(ax, "Person recall by distance (night test set)", "recall")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig3_recall_by_distance.png"), dpi=160); plt.close(fig)

    # ---- table
    lines = ["FINAL COMPARISON - night test set (150 images, 144 people, 108 suitcases)", ""]
    lines.append("%-24s %-30s %8s %8s %8s %8s" % ("pipeline", "model", "P pers", "R pers", "R suit", "mAP50"))
    for p in pipes:
        for tag, name, _ in MODELS:
            lines.append("%-24s %-30s %8.2f %8.2f %8.2f %8.3f" % (
                p, name, val(tag, p, "person", "precision"), val(tag, p, "person", "recall"),
                val(tag, p, "suitcase", "recall"), val(tag, p, "person", "map50")))
    lines += ["", "person recall by distance:"]
    for tag, name, _ in MODELS:
        d = data[tag][1]
        lines.append("  %-30s near %.0f%%  mid %.0f%%  far %.0f%%  (pipeline: %s)" % (
            name, 100 * d.get("near", 0), 100 * d.get("mid", 0), 100 * d.get("far", 0), d.get("_pipeline")))
    open(os.path.join(OUT, "summary_table.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\nCharts saved in", OUT)


if __name__ == "__main__":
    main()
