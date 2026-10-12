"""
app.py - InteliPatrol live monitoring dashboard (Flask web server).

How it works
  * every robot controller writes its live status ("telemetry") to
    %TEMP%\\intelipatrol\\<robot>.json   twice per second, and its latest
    camera frame to  <robot>_camera.jpg
  * this server reads those files and offers them to the web page as JSON
  * the web page (templates/index.html) asks for fresh data every 0.5 s and
    redraws the campus map, robot cards, camera views and alert feed

Run it from the InteliPatrol folder:
    python dashboard/app.py
then open  http://localhost:5000  in a browser.
"""
import glob
import json
import os
import sys
import tempfile
import time

from flask import Flask, abort, jsonify, render_template, send_file, send_from_directory

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "controllers", "patrol_controller"))
import campus_layout as L  # noqa: E402  (same layout file the robot uses)

TELEMETRY_DIR = os.path.join(tempfile.gettempdir(), "vision_security_robot")
OFFLINE_AFTER = 5.0          # seconds without an update -> robot shown as offline

app = Flask(__name__)


def campus_layout():
    """Everything needed to draw the campus map (sent once to the page)."""
    return {
        "half": L.WORLD_HALF,
        "buildings": [{"name": n, "x": c[0], "y": c[1], "w": s[0], "h": s[1], "color": col}
                      for n, c, s, _, col in L.BUILDINGS],
        "roads": [{"x": c[0], "y": c[1], "w": s[0], "h": s[1]} for _, c, s, _ in L.ROADS],
        "sports": {"x": L.SPORTS_GROUND[0][0], "y": L.SPORTS_GROUND[0][1],
                   "w": L.SPORTS_GROUND[1][0], "h": L.SPORTS_GROUND[1][1]},
        "trees": [{"x": x, "y": y} for x, y in L.TREES],
        "benches": [{"x": c[0], "y": c[1], "w": s[0], "h": s[1]} for c, s in L.BENCHES],
        "zones": [{"name": n, "x0": r[0], "x1": r[1], "y0": r[2], "y1": r[3], "restricted": rs}
                  for n, r, rs in L.ZONES],
        "checkpoints": [{"name": n, "x": p[0], "y": p[1]} for n, p in L.CHECKPOINTS],
        "dock": {"name": L.DOCK[0], "x": L.DOCK[1][0], "y": L.DOCK[1][1]},
    }


def read_robots():
    robots = []
    for path in sorted(glob.glob(os.path.join(TELEMETRY_DIR, "*.json"))):
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, ValueError):
            continue                       # file is being rewritten - skip this round
        age = time.time() - os.path.getmtime(path)
        data["online"] = age < OFFLINE_AFTER
        data["age"] = round(age, 1)
        data["has_camera"] = os.path.exists(os.path.join(TELEMETRY_DIR, "%s_camera.jpg" % data["robot"]))
        robots.append(data)
    return robots


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/layout")
def api_layout():
    return jsonify(campus_layout())


@app.route("/api/state")
def api_state():
    robots = read_robots()
    alerts = []
    for r in robots:
        for a in r.get("alerts", []):
            alerts.append(dict(a, robot=r["robot"]))
    alerts.sort(key=lambda a: a["t"], reverse=True)
    return jsonify({"robots": robots, "alerts": alerts[:40], "server_time": time.time()})


@app.route("/camera/<robot>.jpg")
def camera(robot):
    path = os.path.join(TELEMETRY_DIR, "%s_camera.jpg" % os.path.basename(robot))
    if not os.path.exists(path):
        abort(404)
    resp = send_file(path, mimetype="image/jpeg", max_age=0)
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/snapshot/<robot>/<name>")
def snapshot(robot, name):
    """Alert photos live in the robot's run folder (output/run_.../snapshots)."""
    for r in read_robots():
        if r["robot"] == robot and r.get("snapshot_dir"):
            return send_from_directory(r["snapshot_dir"], os.path.basename(name))
    abort(404)


if __name__ == "__main__":
    print("Vision Security Robot dashboard")
    print("  reading telemetry from:", TELEMETRY_DIR)
    print("  open http://localhost:5000 in your browser  (Ctrl+C to stop)")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
