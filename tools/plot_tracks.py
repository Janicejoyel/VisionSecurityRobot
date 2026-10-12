"""
plot_tracks.py - Draw every tracked person's trajectory on the campus map.

Reads output/<run>/tracks.csv (written by the robots while they patrol) and saves
output/<run>/tracks_map.png  plus a short statistics summary (tracks_summary.txt).

Run from the VisionSecurityRobot folder:
    python tools/plot_tracks.py                  (latest run)
    python tools/plot_tracks.py output/run_XXXX
"""
import csv
import glob
import math
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "controllers", "patrol_controller"))
import campus_layout as L  # noqa: E402


def main():
    if len(sys.argv) > 1:
        run = sys.argv[1]
    else:
        runs = [r for r in sorted(glob.glob(os.path.join(ROOT, "output", "run_*")))
                if os.path.exists(os.path.join(r, "tracks.csv"))]
        if not runs:
            sys.exit("No tracks.csv found - run the simulation first.")
        run = runs[-1]
    rows = list(csv.DictReader(open(os.path.join(run, "tracks.csv"))))
    tracks = defaultdict(list)
    for r in rows:
        tracks[(r["robot"], int(r["track_id"]))].append(
            (float(r["sim_time"]), float(r["x"]), float(r["y"]), float(r["speed"]), r["loitering"] == "1"))

    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    h = L.WORLD_HALF
    ax.set_xlim(-h, h); ax.set_ylim(-h, h); ax.set_aspect("equal")
    ax.set_facecolor("#e8efe4")
    for _, (x0, x1, y0, y1), restricted in L.ZONES:
        if restricted:
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, color="#ff4d5e", alpha=0.08))
    for name, (cx, cy), (sx, sy), _, col in L.BUILDINGS:
        ax.add_patch(Rectangle((cx - sx / 2, cy - sy / 2), sx, sy, color=col, ec="k", lw=0.5))
        if sx > 2:
            ax.text(cx, cy, name, ha="center", va="center", fontsize=8)
    for x, y in L.TREES:
        ax.add_patch(Circle((x, y), 0.9, color="#4a8a4a", alpha=0.5))
    for k, (name, (x, y)) in enumerate(L.CHECKPOINTS):
        ax.plot(x, y, "s", color="#e0b000", ms=7)

    cmap = plt.get_cmap("tab10")
    lines = ["MULTI-OBJECT TRACKING SUMMARY", "run: %s" % os.path.basename(run), ""]
    lines.append("%-14s %8s %9s %10s %10s" % ("track", "points", "duration", "max speed", "loitering"))
    for k, ((robot, tid), pts) in enumerate(sorted(tracks.items())):
        pts.sort()
        xs = [p[1] for p in pts]; ys = [p[2] for p in pts]
        col = cmap(k % 10)
        loiter = any(p[4] for p in pts)
        ax.plot(xs, ys, "-", color=col, lw=2, alpha=0.85)
        ax.plot(xs[-1], ys[-1], "o", color=col, ms=6)
        label = "%s#%d" % (robot.replace("patrolbot_", "R"), tid)
        ax.annotate(label + (" (loitering)" if loiter else ""), (xs[-1], ys[-1]), fontsize=7,
                    xytext=(4, 4), textcoords="offset points", color=col)
        if loiter:
            ax.add_patch(Circle((sum(xs) / len(xs), sum(ys) / len(ys)), 2.5, fill=False, ls="--", ec="red"))
        lines.append("%-14s %8d %8.0fs %8.2fm/s %10s" % (label, len(pts), pts[-1][0] - pts[0][0],
                                                        max(p[3] for p in pts), "YES" if loiter else "-"))
    ax.set_title("Tracked people - trajectories in campus coordinates")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    fig.tight_layout()
    fig.savefig(os.path.join(run, "tracks_map.png"), dpi=140)
    lines += ["", "tracks: %d   (each robot keeps its own track ids)" % len(tracks)]
    open(os.path.join(run, "tracks_summary.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\nSaved", os.path.join(run, "tracks_map.png"))


if __name__ == "__main__":
    main()
