"""
patrol_controller.py - The "brain" of an InteliPatrol robot.

Sense -> Think -> Act, every simulation step:

  SENSE : GPS (position), InertialUnit (heading), LiDAR (distances around),
          camera + YOLOv8 (people, bags) fused with LiDAR for position
  THINK : - update the cost map with obstacles the LiDAR sees
          - A* path planning to the next checkpoint (re-plan when blocked)
          - threat scoring of everything the camera sees (threat.py)
          - behaviour state machine:
              PATROL -> SCAN -> PATROL ...               (normal duty)
              PATROL/SCAN -> INVESTIGATE -> ALERT        (HIGH threat seen)
              any    -> AVOID -> previous state          (emergency obstacle)
              PATROL -> RETURN_TO_BASE -> CHARGING       (battery low)
  ACT   : differential-drive wheel speeds to follow the path

Telemetry (position, state, battery, path, events) is written to a JSON file
for the web dashboard.
"""
import json
import math
import os
import sys
import tempfile
import time

from controller import Robot

import campus_layout as L
import vision as V
from costmap import CostMap
from planner import astar, path_length
from threat import CampusClock, ThreatAssessor, HIGH, MEDIUM
from tracker import MapTracker, colour_histogram

# ----------------------------------------------------------------- settings
WHEEL_RADIUS = 0.10        # m
TRACK_WIDTH = 0.60         # effective track for skid steering (m)
CRUISE_SPEED = 0.7         # m/s
MAX_TURN_RATE = 1.5        # rad/s
HEADING_GAIN = 2.2
LOOKAHEAD = 1.0            # m - how far ahead on the path to aim
ARRIVE_RADIUS = 0.6        # m - "checkpoint reached"

FRONT_CONE = math.radians(25)
EMERGENCY_DIST = 0.55      # m - obstacle this close in front -> AVOID
SLOW_DIST = 1.8            # m - start slowing down
MARK_RANGE = 4.5           # m - LiDAR hits closer than this go into the map
REPLAN_CHECK_EVERY = 0.5   # s
YIELD_TIME = 4.0           # s - wait this long for a blocked path to clear
PEER_CHECK_EVERY = 1.0     # s - how often to read the other robots' status
PEER_TIMEOUT = 5.0         # s - a robot silent this long is treated as offline

SCAN_AT_CHECKPOINT = True  # rotate 360 degrees at each checkpoint
SCAN_TURN_RATE = 0.9       # rad/s

BATTERY_DRAIN_MOVING = 0.20   # % per second while driving
BATTERY_DRAIN_IDLE = 0.02     # % per second while stopped
BATTERY_CHARGE_RATE = 3.0     # % per second on the dock
BATTERY_LOW = 25.0            # % -> go home
BATTERY_FULL = 98.0           # % -> resume patrol

STUCK_WINDOW = 6.0         # s
STUCK_DISTANCE = 0.25      # m

TELEMETRY_EVERY = 0.5      # s
TELEMETRY_DIR = os.path.join(tempfile.gettempdir(), "vision_security_robot")

# ---- vision (Layer 2)
VISION_ENABLED = True
DETECT_EVERY = 0.512       # s between YOLO runs (keeps the simulation fast)
CAMERA_OFFSET = 0.24       # m - camera is this far in front of the robot centre
INVESTIGATE_STANDOFF = 4.0  # m - stop this far from the target
INVESTIGATE_TIMEOUT = 50.0  # s - give up after this long
CONFIRM_WINDOW = 5.0       # s - time allowed to re-detect the target up close
CONFIRMATIONS_NEEDED = 2
ALERT_PAUSE = 3.0          # s - robot holds position while raising an alert
MAX_ALERT_RANGE = 12.0     # m - farther detections are too inaccurate to act on
OBJECT_CHECK_EVERY = 1.0   # s - how often to look for new unmapped objects

# ---- FOCV: low-light enhancement + frame recording
# Enhancement mode is chosen from our evaluation (results/*/summary.txt):
#   pretrained YOLOv8n  : enhancement helps slightly  -> "auto"
#   campus-trained model: trained on raw night images, enhancement HURTS (mAP 0.912 -> 0.891) -> "off"
ENHANCE_MODE = "auto"      # "auto" = enhance only dark images, "on" = always, "off" = never
ENHANCE_PIPELINE = "bilateral+gamma+CLAHE"   # any name from vision.PIPELINES
DARK_THRESHOLD = 90        # mean brightness (0-255) below which an image counts as dark
SAVE_FRAMES_EVERY = 3.0    # s - save raw camera frames for offline evaluation (0 = off)
MAX_SAVED_FRAMES = 300
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "yolov8n.pt")            # COCO-pretrained
CUSTOM_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "campus_yolo.pt")  # trained on our dataset
USE_CUSTOM_MODEL = True    # use campus_yolo.pt when it exists
# every run gets its own folder:  output/run_YYYY-MM-DD_HH-MM-SS/
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", time.strftime("run_%Y-%m-%d_%H-%M-%S"))
SNAPSHOT_DIR = os.path.join(OUTPUT_DIR, "snapshots")


def wrap(a):
    """Wrap an angle to [-pi, pi]."""
    return (a + math.pi) % (2 * math.pi) - math.pi


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class PatrolBot:
    def __init__(self):
        self.robot = Robot()
        self.dt_ms = int(self.robot.getBasicTimeStep())
        self.dt = self.dt_ms / 1000.0
        self.name = self.robot.getName()

        # ---- actuators
        self.left = [self.robot.getDevice(n) for n in ("fl_motor", "rl_motor")]
        self.right = [self.robot.getDevice(n) for n in ("fr_motor", "rr_motor")]
        for m in self.left + self.right:
            m.setPosition(float("inf"))      # velocity-control mode
            m.setVelocity(0.0)
        self.max_wheel = self.left[0].getMaxVelocity()

        # ---- sensors
        self.gps = self.robot.getDevice("gps")
        self.gps.enable(self.dt_ms)
        self.imu = self.robot.getDevice("imu")
        self.imu.enable(self.dt_ms)
        self.lidar = self.robot.getDevice("lidar")
        self.lidar.enable(self.dt_ms)
        self.lidar_res = self.lidar.getHorizontalResolution()
        self.lidar_fov = self.lidar.getFov()
        self.lidar_max = self.lidar.getMaxRange()
        self.ranges = []

        # ---- vision
        self.camera = self.robot.getDevice("camera")
        period = max(self.dt_ms, int(round(DETECT_EVERY * 1000 / self.dt_ms)) * self.dt_ms)
        self.cam = {"width": self.camera.getWidth(), "height": self.camera.getHeight(),
                    "hfov": self.camera.getFov(), "offset": CAMERA_OFFSET}
        self.detector = None
        if VISION_ENABLED:
            self.camera.enable(period)
            if USE_CUSTOM_MODEL and os.path.exists(CUSTOM_MODEL_PATH):
                model = CUSTOM_MODEL_PATH
            else:
                model = MODEL_PATH if os.path.exists(MODEL_PATH) else "yolov8n.pt"
            self.detector = V.Detector(model)
            self.model_name = os.path.basename(model)
        self.clock = CampusClock()
        self.assessor = ThreatAssessor(self.clock)
        self.last_detect = -1.0
        self.last_dets = []
        self.alerts = []
        self.alert_count = 0
        self.target = None
        self.inv = {}
        self.recent_people = []    # (x, y, t) of people seen in the last 10 s
        self.last_object_check = 0.0
        self.vision_stats = {}
        self.tracker = MapTracker(labels=("person",))
        self.last_track_log = -1e9
        self.last_frame_save = -1e9
        self.frames_saved = 0

        # ---- brain
        self.map = CostMap()
        self.map.obs_dt = self.dt
        self.own_route = self._parse_route()          # this robot's sector
        self.route = list(self.own_route)              # may grow when covering a peer
        self.route_idx = 0
        self.dock = L.DOCKS[int(self._arg("dock", "0"))]
        self.priority = int(self._arg("priority", "1"))
        self.yield_time = YIELD_TIME + 2.0 * (2 - min(self.priority, 2))   # robot 1 waits longer
        self.peers = {}                                # name -> latest telemetry
        self.covering = []                             # names of peers whose sector we cover
        self.last_peer_check = 0.0
        self.state = "PATROL"
        self.prev_state = None
        self.path, self.path_idx = None, 0
        self.goal_name, self.goal_xy = None, None
        self.battery = float(self._arg("battery", "100"))
        self.events = []
        self.x = self.y = self.yaw = 0.0
        self.front_min = math.inf
        self.state_t0 = 0.0
        self.scan_turned, self.scan_last_yaw = 0.0, 0.0
        self.avoid_turn_dir = 1.0
        self.stuck_ref = (0.0, 0.0, 0.0)
        self.last_replan_check = 0.0
        self.last_telemetry = -1.0
        self.map_version_seen = self.map.version
        self.blocked_since = None
        self.distance_travelled = 0.0
        self.have_fix = False
        self.moving = False
        self.map_dirty, self.last_rebuild = False, 0.0
        self.checkpoints_visited = 0
        os.makedirs(TELEMETRY_DIR, exist_ok=True)

    # ------------------------------------------------------------ utilities
    def _arg(self, key, default=None):
        for arg in sys.argv[1:]:
            if arg.startswith("--%s=" % key):
                return arg.split("=", 1)[1]
        return default

    def _parse_route(self):
        """controllerArgs:  --route=0,1,2   (checkpoint indices this robot owns)."""
        r = self._arg("route")
        if r:
            return [int(k) for k in r.split(",") if k.strip()]
        return list(range(len(L.CHECKPOINTS)))

    def now(self):
        return self.robot.getTime()

    def log(self, kind, msg):
        t = self.now()
        stamp = "%02d:%02d" % (int(t) // 60, int(t) % 60)
        print("[%s %s] %-10s %s" % (self.name, stamp, kind, msg))
        self.events.append({"t": round(t, 1), "type": kind, "msg": msg,
                            "x": round(self.x, 2), "y": round(self.y, 2)})
        self.events = self.events[-50:]

    def set_state(self, new):
        if new != self.state:
            self.log("STATE", "%s -> %s" % (self.state, new))
            self.state = new
            self.state_t0 = self.now()
            self.stuck_ref = (self.x, self.y, self.now())
            self.blocked_since = None

    def drive(self, v, w):
        """v = forward speed (m/s), w = turn rate (rad/s, +ve = left)."""
        wl = (v - w * TRACK_WIDTH / 2) / WHEEL_RADIUS
        wr = (v + w * TRACK_WIDTH / 2) / WHEEL_RADIUS
        k = max(1.0, abs(wl) / self.max_wheel, abs(wr) / self.max_wheel)
        for m in self.left:
            m.setVelocity(wl / k)
        for m in self.right:
            m.setVelocity(wr / k)
        self.moving = abs(v) > 0.01 or abs(w) > 0.01

    # --------------------------------------------------------------- SENSE
    def sense(self):
        px, py = self.x, self.y
        gx, gy, _ = self.gps.getValues()
        if not (math.isnan(gx) or math.isnan(gy)):
            self.x, self.y = gx, gy
            if self.have_fix:
                self.distance_travelled += math.hypot(self.x - px, self.y - py)
            self.have_fix = True
        self.yaw = self.imu.getRollPitchYaw()[2]

        ranges = self.lidar.getRangeImage()
        self.ranges = ranges
        n = len(ranges)
        step = self.lidar_fov / max(1, n - 1)
        front, hits = math.inf, []
        for k, r in enumerate(ranges):
            if math.isinf(r) or math.isnan(r) or r >= self.lidar_max * 0.98:
                continue
            a = self.lidar_fov / 2 - k * step          # index 0 = leftmost ray
            if abs(a) <= FRONT_CONE:
                front = min(front, r)
            if r <= MARK_RANGE:
                ga = self.yaw + a
                hits.append((self.x + r * math.cos(ga), self.y + r * math.sin(ga)))
        self.front_min = front

        # left vs right free space (used to pick an escape direction)
        half = n // 2
        self.left_space = sum(min(r, 3.0) for r in ranges[:half])
        self.right_space = sum(min(r, 3.0) for r in ranges[half:])

        # update the map with what we see
        t = self.now()
        if self.map.mark_obstacles(hits, t):
            self.map_dirty = True
        if self.map.forget_old(t):
            self.map_dirty = True
        if self.map_dirty and t - self.last_rebuild >= REPLAN_CHECK_EVERY:
            self.map.rebuild()
            self.map_dirty, self.last_rebuild = False, t

    # ---------------------------------------------------------------- THINK
    def plan_to(self, name, xy):
        self.goal_name, self.goal_xy = name, xy
        path, expanded = astar(self.map, (self.x, self.y), xy)
        if path is None:
            self.path = None
            self.log("PLAN", "no path to %s (expanded %d cells)" % (name, expanded))
            return False
        self.path, self.path_idx = path, 0
        self.log("PLAN", "A* -> %s: %.1f m, %d waypoints, %d cells expanded"
                 % (name, path_length(path), len(path), expanded))
        return True

    def current_goal(self):
        if self.state == "RETURN_TO_BASE":
            return self.dock
        if self.state == "INVESTIGATE" and self.inv.get("approach"):
            return ("investigation point", self.inv["approach"])
        return L.CHECKPOINTS[self.route[self.route_idx]]

    def check_replan(self):
        """Re-plan if the map changed and our path now crosses an obstacle."""
        t = self.now()
        if t - self.last_replan_check < REPLAN_CHECK_EVERY:
            return
        self.last_replan_check = t
        blocked = bool(self.path) and self.map.path_blocked(self.path, self.path_idx)
        if not blocked:
            self.blocked_since = None
            return
        # Something is in the way. A walking person usually clears the path in a
        # few seconds, so first WAIT; only re-plan if it is still blocked.
        if self.blocked_since is None:
            self.blocked_since = t
            self.log("YIELD", "path blocked - waiting %.0f s for it to clear" % self.yield_time)
        elif t - self.blocked_since >= self.yield_time:
            self.blocked_since = None
            self.log("REPLAN", "path still blocked - re-planning around the obstacle")
            self.plan_to(*self.current_goal())

    def follow_path(self):
        """Pure-pursuit style path following. Returns (v, w, arrived)."""
        path = self.path
        while (self.path_idx < len(path) - 1 and
               math.hypot(path[self.path_idx][0] - self.x,
                          path[self.path_idx][1] - self.y) < LOOKAHEAD):
            self.path_idx += 1
        tx, ty = path[self.path_idx]
        gx, gy = path[-1]
        d_goal = math.hypot(gx - self.x, gy - self.y)
        if self.path_idx == len(path) - 1 and d_goal < ARRIVE_RADIUS:
            return 0.0, 0.0, True

        err = wrap(math.atan2(ty - self.y, tx - self.x) - self.yaw)
        if abs(err) > 0.9:                                   # turn on the spot
            return 0.0, math.copysign(1.2, err), False
        v = CRUISE_SPEED * max(0.25, 1.0 - abs(err) / 0.9)
        if self.front_min < SLOW_DIST:                       # obstacle ahead
            v *= clamp((self.front_min - EMERGENCY_DIST) / (SLOW_DIST - EMERGENCY_DIST), 0.3, 1.0)
        if d_goal < 1.5:                                     # final approach
            v *= max(0.3, d_goal / 1.5)
        w = clamp(HEADING_GAIN * err, -MAX_TURN_RATE, MAX_TURN_RATE)
        return v, w, False

    def is_stuck(self):
        t = self.now()
        rx, ry, rt = self.stuck_ref
        if t - rt >= STUCK_WINDOW:
            moved = math.hypot(self.x - rx, self.y - ry)
            self.stuck_ref = (self.x, self.y, t)
            return moved < STUCK_DISTANCE
        return False

    def update_battery(self):
        if self.state == "CHARGING":
            self.battery = min(100.0, self.battery + BATTERY_CHARGE_RATE * self.dt)
        else:
            rate = BATTERY_DRAIN_MOVING if self.moving else BATTERY_DRAIN_IDLE
            self.battery = max(0.0, self.battery - rate * self.dt)

    # ------------------------------------------------------ state handlers
    def guard(self):
        """Emergency checks while driving. Returns True if we switched to AVOID."""
        if self.front_min < EMERGENCY_DIST or self.is_stuck():
            self.log("AVOID", "obstacle %.2f m ahead - evasive manoeuvre" % self.front_min
                     if self.front_min < EMERGENCY_DIST else "robot stuck - recovery manoeuvre")
            self.prev_state = self.state
            self.avoid_turn_dir = 1.0 if self.left_space >= self.right_space else -1.0
            self.set_state("AVOID")
            return True
        return False

    def do_navigate(self):
        """Shared by PATROL and RETURN_TO_BASE."""
        if self.guard():
            return 0.0, 0.0

        if self.path is None:
            if not self.plan_to(*self.current_goal()):
                if self.state == "PATROL":             # skip unreachable checkpoint
                    self.route_idx = (self.route_idx + 1) % len(self.route)
                return 0.0, 0.0
        self.check_replan()
        if self.path is None or self.blocked_since is not None:
            return 0.0, 0.0

        v, w, arrived = self.follow_path()
        if arrived:
            self.path = None
            if self.state == "RETURN_TO_BASE":
                self.log("DOCK", "docked at charging station (battery %.0f%%)" % self.battery)
                self.set_state("CHARGING")
            else:
                name = self.goal_name
                self.checkpoints_visited += 1
                self.log("CHECKPOINT", "reached %s" % name)
                self.route_idx = (self.route_idx + 1) % len(self.route)
                if SCAN_AT_CHECKPOINT:
                    self.scan_turned, self.scan_last_yaw = 0.0, self.yaw
                    self.set_state("SCAN")
            return 0.0, 0.0
        return v, w

    def do_scan(self):
        self.scan_turned += abs(wrap(self.yaw - self.scan_last_yaw))
        self.scan_last_yaw = self.yaw
        if self.scan_turned >= 2 * math.pi or self.now() - self.state_t0 > 25:
            self.log("SCAN", "360-degree scan complete")
            self.set_state("PATROL")
            return 0.0, 0.0
        return 0.0, SCAN_TURN_RATE

    def do_avoid(self):
        t = self.now() - self.state_t0
        if t < 1.0:                                  # 1) back up
            return -0.25, 0.0
        if t < 3.5 and self.front_min < 1.3:         # 2) turn to the freer side
            return 0.0, self.avoid_turn_dir * 1.0
        # 3) clear -> force a rebuild + re-plan and go back to what we were doing
        self.map.rebuild()
        self.path = None
        self.stuck_ref = (self.x, self.y, self.now())
        self.set_state(self.prev_state or "PATROL")
        return 0.0, 0.0

    def do_charging(self):
        if self.battery >= BATTERY_FULL:
            self.log("BATTERY", "charged to %.0f%% - resuming patrol" % self.battery)
            self.path = None
            self.set_state("PATROL")
        return 0.0, 0.0

    # ------------------------------------------------------------ vision
    def perceive(self):
        """Run YOLO on the camera image, locate detections, score threats."""
        t = self.now()
        if (not self.detector or not self.detector.ok or
                t - self.last_detect < DETECT_EVERY or
                self.state not in ("PATROL", "SCAN", "INVESTIGATE")):
            return
        self.last_detect = t
        raw = V.camera_frame(self.camera)
        self.save_raw_frame(raw, t)
        b = V.brightness(raw)
        mode = "off" if (self.detector and self.detector.custom) else ENHANCE_MODE
        if mode == "on" or (mode == "auto" and b < DARK_THRESHOLD):
            frame, info = V.run_pipeline(raw, ENHANCE_PIPELINE)
            info["enhanced"] = ENHANCE_PIPELINE != "raw"
        else:
            frame, info = raw, {"brightness_before": round(b, 1), "enhanced": False}
        self.vision_stats = info
        dets = self.detector.detect(frame)
        pose = (self.x, self.y, self.yaw)
        for d in dets:
            d["pos"] = V.locate(d, pose, self.cam, self.ranges, self.lidar_fov)
        # multi-object tracking (map coordinates + clothing colour)
        for d in dets:
            if d["label"] == "person":
                d["hist"] = colour_histogram(frame, d["box"])
        self.tracker.update([d for d in dets if d["pos"]["dist"] <= MAX_ALERT_RANGE + 3], t)
        self.log_tracks(t)
        self.last_dets = dets
        self.last_frame = frame
        self.recent_people = [p for p in self.recent_people if t - p[2] < 30.0]
        self.recent_people += [(d["pos"]["x"], d["pos"]["y"], t) for d in dets if d["label"] == "person"]

        # live camera view for the dashboard
        tag = ("ENHANCED g=%.2f" % info["gamma"]) if info.get("enhanced") else "RAW"
        live = V.annotate(frame, dets, "%s | %s | %s | %s" % (self.name, self.clock.text(t), self.state, tag))
        V.save_image(live, os.path.join(TELEMETRY_DIR, "%s_camera.jpg" % self.name))

        if self.state == "INVESTIGATE":
            self.track_target([d for d in dets if d["pos"]["dist"] <= MAX_ALERT_RANGE + 3])
            return

        near = [d for d in dets if d["pos"]["dist"] <= MAX_ALERT_RANGE]
        threats = self.assessor.assess(near, t)
        for trk in self.tracker.confirmed():                 # behaviour analysis
            if trk.loitering and not trk.loiter_reported:
                trk.loiter_reported = True
                self.log("TRACK", "person #%d is loitering at (%.1f, %.1f)" % (trk.id, trk.x, trk.y))
                threats.append(self.assessor.loitering(trk, t))
        threats.sort(key=lambda th: -th["score"])
        for th in threats:
            if not self.assessor.is_new(th, t):
                continue
            self.assessor.remember(th, t)
            if th["score"] >= HIGH:
                self.start_investigation(th, frame)
                return
            if th["score"] >= MEDIUM:
                self.raise_alert(th, frame, confirmed=False)
            else:
                self.log("SEEN", "%s at (%.1f, %.1f), score %d - no action"
                         % (th["kind"].lower(), th["x"], th["y"], th["score"]))

    def log_tracks(self, t):
        """Append confirmed tracks to output/<run>/tracks.csv every 2 s (for plots)."""
        if t - self.last_track_log < 2.0:
            return
        self.last_track_log = t
        trks = self.tracker.confirmed()
        if not trks:
            return
        path = os.path.join(OUTPUT_DIR, "tracks.csv")
        try:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            new = not os.path.exists(path)
            with open(path, "a") as f:
                if new:
                    f.write("sim_time,robot,track_id,x,y,speed,unseen,loitering\n")
                for trk in trks:
                    f.write("%.1f,%s,%d,%.2f,%.2f,%.2f,%.1f,%d\n" % (
                        t, self.name, trk.id, trk.x, trk.y, trk.speed, t - trk.last_seen, trk.loitering))
        except OSError:
            pass

    def save_raw_frame(self, raw, t):
        """Keep raw (un-enhanced) frames so the enhancement can be evaluated offline."""
        if (not SAVE_FRAMES_EVERY or raw is None or self.frames_saved >= MAX_SAVED_FRAMES or
                t - self.last_frame_save < SAVE_FRAMES_EVERY):
            return
        self.last_frame_save = t
        path = os.path.join(OUTPUT_DIR, "frames_raw", "%s_%06.1f.jpg" % (self.name, t))
        if V.save_image(raw, path):
            self.frames_saved += 1

    def check_unknown_objects(self):
        """Map change detection: has a small object appeared that is not on the map?"""
        t = self.now()
        if (t - self.last_object_check < OBJECT_CHECK_EVERY or
                self.state not in ("PATROL", "SCAN")):
            return
        self.last_object_check = t
        for x, y, size in self.map.small_unknown_objects():
            if math.hypot(x - self.x, y - self.y) > 8.0:
                continue
            if any(math.hypot(px - x, py - y) < 2.5 for px, py, _ in self.recent_people):
                continue                               # probably a person, not an object
            if any(math.hypot(p["x"] - x, p["y"] - y) < 2.5 for p in self.peers.values()):
                continue                               # that's another InteliPatrol robot
            th = self.assessor.unknown_object(x, y, size, t)
            if not self.assessor.is_new(th, t):
                continue
            self.assessor.remember(th, t)
            self.log("LIDAR", "map change: unmapped %.1f m object at (%.1f, %.1f) in %s"
                     % (size, x, y, th["zone"]))
            if th["score"] >= HIGH:
                self.start_investigation(th, getattr(self, "last_frame", None))
            else:
                self.raise_alert(th, getattr(self, "last_frame", None), confirmed=False)
            return

    def lidar_confirms(self, tg):
        """Is there still a static unmapped obstacle where the target was?"""
        now = self.now()
        for c, v in self.map.dynamic.items():
            x, y = self.map.cell_to_world(*c)
            if (math.hypot(x - tg["x"], y - tg["y"]) < 1.0 and now - v[1] < 3.0
                    and self.map.is_persistent(v)):
                return True
        return False

    def _matches(self, d):
        kind = self.target["kind"]
        if kind in ("UNATTENDED_OBJECT", "BAG", "UNKNOWN_OBJECT"):
            return d["label"] in V.BAG_LABELS
        return d["label"] == "person"

    def track_target(self, dets):
        """Keep the target position up to date (people move!) and count confirmations."""
        tg = self.target
        if tg["kind"] == "UNKNOWN_OBJECT" and self.inv["phase"] == "confirm":
            if any(d["label"] == "person" and
                   math.hypot(d["pos"]["x"] - tg["x"], d["pos"]["y"] - tg["y"]) < 1.5 for d in dets):
                self.inv["person_seen"] = True
        best = None
        for d in dets:
            if not self._matches(d):
                continue
            dd = math.hypot(d["pos"]["x"] - tg["x"], d["pos"]["y"] - tg["y"])
            if dd < 4.0 and (best is None or dd < best[0]):
                best = (dd, d)
        if best is None:
            return
        d = best[1]
        tg["x"] = round(0.5 * tg["x"] + 0.5 * d["pos"]["x"], 2)     # smooth the estimate
        tg["y"] = round(0.5 * tg["y"] + 0.5 * d["pos"]["y"], 2)
        self.inv["last_seen"] = self.now()
        if self.inv["phase"] == "confirm":
            self.inv["confirmations"] += 1
            self.inv["best_conf"] = max(self.inv["best_conf"], d["conf"])
            if tg["kind"] == "UNKNOWN_OBJECT":           # camera recognised what it is
                tg["kind"] = "UNATTENDED_OBJECT"
                tg["reasons"] = tg["reasons"] + ["camera classified it as a %s" % d["label"]]

    def start_investigation(self, th, frame):
        self.target = th
        self.inv = {"phase": "approach", "confirmations": 0, "best_conf": 0.0,
                    "last_seen": self.now(), "approach": None, "planned_for": None,
                    "last_plan": -10.0, "first_frame": frame}
        self.log("THREAT", "%s score %d (%s) at (%.1f, %.1f) in %s - %s"
                 % (th["level"], th["score"], th["kind"], th["x"], th["y"], th["zone"],
                    "; ".join(th["reasons"])))
        self.path = None
        self.set_state("INVESTIGATE")

    def do_investigate(self):
        t = self.now() - self.state_t0
        tg, inv = self.target, self.inv
        if t > INVESTIGATE_TIMEOUT:
            return self.finish_investigation(confirmed=inv["confirmations"] > 0)

        dx, dy = tg["x"] - self.x, tg["y"] - self.y
        dist = math.hypot(dx, dy)

        if inv["phase"] == "approach":
            if dist <= INVESTIGATE_STANDOFF + 0.5:
                inv["phase"], inv["confirm_t0"] = "confirm", self.now()
                self.path = None
                self.log("INVESTIGATE", "in position %.1f m from target - confirming" % dist)
                return 0.0, 0.0
            if self.guard():
                return 0.0, 0.0
            k = (dist - INVESTIGATE_STANDOFF) / dist
            approach = (self.x + dx * k, self.y + dy * k)
            moved = (inv["planned_for"] is None or
                     math.dist(inv["planned_for"], (tg["x"], tg["y"])) > 1.5)
            if self.path is None or (moved and self.now() - inv["last_plan"] > 2.0):
                inv["approach"], inv["planned_for"] = approach, (tg["x"], tg["y"])
                inv["last_plan"] = self.now()
                if not self.plan_to("investigation point", approach):
                    return self.finish_investigation(confirmed=False)
            self.check_replan()
            if self.path is None or self.blocked_since is not None:
                return 0.0, 0.0
            v, w, arrived = self.follow_path()
            if arrived:
                inv["phase"], inv["confirm_t0"] = "confirm", self.now()
                self.path = None
                return 0.0, 0.0
            return v, w

        # phase "confirm": face the target and look again
        if inv["confirmations"] >= CONFIRMATIONS_NEEDED:
            return self.finish_investigation(confirmed=True)
        if self.now() - inv["confirm_t0"] > CONFIRM_WINDOW:
            if (tg["kind"] == "UNKNOWN_OBJECT" and not inv.get("person_seen")
                    and self.lidar_confirms(tg)):
                # LiDAR still sees it but the camera AI cannot say what it is
                tg["kind"] = "UNIDENTIFIED_OBJECT"
                tg["reasons"] = tg["reasons"] + [
                    "LiDAR re-confirmed it; camera AI could not classify it - needs human review"]
                inv["best_conf"] = 0.0
                return self.finish_investigation(confirmed=True)
            if inv.get("person_seen"):
                self.log("INVESTIGATE", "it was a person, not an object - dismissed")
            return self.finish_investigation(confirmed=False)
        err = wrap(math.atan2(dy, dx) - self.yaw)
        if abs(err) > 0.12:
            return 0.0, clamp(2.0 * err, -1.0, 1.0)
        return 0.0, 0.0

    def finish_investigation(self, confirmed):
        tg = self.target
        if confirmed:
            if self.inv["best_conf"] > 0:
                tg["reasons"] = tg["reasons"] + ["confirmed up close (conf %.2f)" % self.inv["best_conf"]]
            self.raise_alert(tg, getattr(self, "last_frame", None), confirmed=True)
            self.set_state("ALERT")
        else:
            self.log("INVESTIGATE", "target not confirmed - logged as a possible false alarm")
            self.raise_alert(dict(tg, level="LOW", kind=tg["kind"] + "_UNCONFIRMED"),
                             self.inv.get("first_frame"), confirmed=False, quiet=True)
            self.set_state("PATROL")
        self.path = None
        return 0.0, 0.0

    def do_alert(self):
        if self.now() - self.state_t0 >= ALERT_PAUSE:
            self.target = None
            self.set_state("PATROL")
        return 0.0, 0.0

    def raise_alert(self, th, frame, confirmed, quiet=False):
        """Save an annotated snapshot + CSV row and add it to the alert list."""
        t = self.now()
        self.alert_count += 1
        clock = self.clock.text(t)
        dets = th.get("detections") or self.last_dets
        header = "InteliPatrol %s | %s | %s %s score %d%s" % (
            self.name, clock, th["level"], th["kind"], th["score"], " CONFIRMED" if confirmed else "")
        footer = "%s (%.1f, %.1f) | %s" % (th["zone"], th["x"], th["y"], "; ".join(th["reasons"]))
        img = V.annotate(frame if frame is not None else getattr(self, "last_frame", None),
                         self.last_dets if confirmed else dets, header, footer)
        fname = "%s_alert_%03d_%s_%s.jpg" % (self.name.replace("patrolbot_", "R"), self.alert_count,
                                            th["kind"].lower(), clock.replace(":", ""))
        saved = V.save_image(img, os.path.join(SNAPSHOT_DIR, fname))
        alert = {"id": self.alert_count, "t": round(t, 1), "campus_time": clock,
                 "level": th["level"], "kind": th["kind"], "score": th["score"],
                 "x": th["x"], "y": th["y"], "zone": th["zone"], "confirmed": confirmed,
                 "reasons": th["reasons"], "snapshot": fname if saved else None}
        self.alerts.append(alert)
        self.alerts = self.alerts[-30:]
        self._append_csv(alert)
        if not quiet:
            self.log("ALERT", "%s %s score %d at (%.1f, %.1f) %s%s -> %s"
                     % (th["level"], th["kind"], th["score"], th["x"], th["y"], th["zone"],
                        " [CONFIRMED]" if confirmed else "", fname if saved else "no snapshot"))

    def _append_csv(self, a):
        path = os.path.join(OUTPUT_DIR, "alerts.csv")
        try:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            new = not os.path.exists(path)
            with open(path, "a") as f:
                if new:
                    f.write("id,sim_time,campus_time,robot,level,kind,score,x,y,zone,confirmed,reasons,snapshot\n")
                f.write('%d,%.1f,%s,%s,%s,%s,%d,%.2f,%.2f,%s,%s,"%s",%s\n' % (
                    a["id"], a["t"], a["campus_time"], self.name, a["level"], a["kind"], a["score"],
                    a["x"], a["y"], a["zone"], a["confirmed"], "; ".join(a["reasons"]),
                    a["snapshot"] or ""))
        except OSError:
            pass

    # ------------------------------------------------------- multi-robot
    def coordinate(self):
        """Read the other robots' telemetry files and react:
          * task reallocation - cover a peer's checkpoints while it charges/is offline
          * shared alert memory - don't re-investigate a threat a peer already handled"""
        t = self.now()
        if t - self.last_peer_check < PEER_CHECK_EVERY:
            return
        self.last_peer_check = t
        peers = {}
        for fn in os.listdir(TELEMETRY_DIR):
            if not fn.endswith(".json") or fn == "%s.json" % self.name:
                continue
            path = os.path.join(TELEMETRY_DIR, fn)
            try:
                if time.time() - os.path.getmtime(path) > PEER_TIMEOUT:
                    continue                                  # offline / old run
                with open(path) as f:
                    d = json.load(f)
            except (OSError, ValueError):
                continue
            if abs(d.get("time", 0) - t) > 30:               # telemetry from another run
                continue
            peers[d["robot"]] = d
        self.peers = peers

        # shared alert memory
        for p in peers.values():
            for a in p.get("alerts", []):
                if not a["kind"].endswith("_UNCONFIRMED"):
                    entry = (a["kind"], a["x"], a["y"], a["t"])
                    if entry not in self.assessor.memory:
                        self.assessor.memory.append(entry)

        # task reallocation
        need = [n for n, p in peers.items() if p.get("state") in ("RETURN_TO_BASE", "CHARGING")]
        if self.state in ("RETURN_TO_BASE", "CHARGING"):
            need = []                                         # we are busy ourselves
        if need != self.covering:
            route = list(self.own_route)
            for n in need:
                route += [k for k in peers[n].get("own_route", []) if k not in route]
            current = self.route[self.route_idx] if self.route else None
            for n in need:
                if n not in self.covering:
                    self.log("COORD", "%s is charging - taking over its checkpoints (%s)" % (
                        n, ", ".join(L.CHECKPOINTS[k][0] for k in peers[n].get("own_route", []))))
            for n in self.covering:
                if n not in need:
                    self.log("COORD", "%s is back on patrol - returning to my own sector" % n)
            newly = [n for n in need if n not in self.covering]
            self.covering = need
            self.route = route
            self.route_idx = route.index(current) if current in route else 0
            if newly:
                # head straight for the nearest checkpoint of the sector nobody is watching
                extra = [k for n in newly for k in peers[n].get("own_route", [])]
                k = min(extra, key=lambda c: math.dist(L.CHECKPOINTS[c][1], (self.x, self.y)))
                self.route_idx = route.index(k)
                if self.state == "PATROL":
                    self.path = None
                    self.log("COORD", "heading to %s first" % L.CHECKPOINTS[k][0])
            elif self.state == "PATROL" and current not in route:
                self.path = None                      # re-plan to a checkpoint we still own

    # ------------------------------------------------------------ telemetry
    def write_telemetry(self):
        t = self.now()
        if t - self.last_telemetry < TELEMETRY_EVERY:
            return
        self.last_telemetry = t
        goal = self.current_goal() if self.state not in ("CHARGING",) else self.dock
        data = {
            "robot": self.name, "time": round(t, 1),
            "x": round(self.x, 3), "y": round(self.y, 3), "yaw": round(self.yaw, 3),
            "state": self.state, "battery": round(self.battery, 1),
            "goal": {"name": goal[0], "x": goal[1][0], "y": goal[1][1]},
            "path": [[round(px, 2), round(py, 2)] for px, py in (self.path or [])[self.path_idx:]],
            "dynamic_obstacles": [list(self.map.cell_to_world(i, j)) for (i, j) in self.map.dynamic],
            "distance_travelled": round(self.distance_travelled, 1),
            "checkpoints_visited": self.checkpoints_visited,
            "own_route": self.own_route,
            "route": self.route,
            "covering": self.covering,
            "dock": {"name": self.dock[0], "x": self.dock[1][0], "y": self.dock[1][1]},
            "priority": self.priority,
            "events": self.events[-20:],
            "campus_time": self.clock.text(t),
            "night": self.clock.is_night(t),
            "model": getattr(self, "model_name", None),
            "vision": ("ok" if self.detector and self.detector.ok else
                       (self.detector.error if self.detector else "disabled")),
            "tracks": self.tracker.summary(t),
            "detections": [{"label": d["label"], "conf": d["conf"], "x": d["pos"]["x"], "track_id": d.get("track_id"),
                            "y": d["pos"]["y"], "dist": d["pos"]["dist"]} for d in self.last_dets],
            "target": ({k: self.target[k] for k in ("kind", "score", "level", "x", "y", "zone")}
                       if self.target else None),
            "alerts": self.alerts[-20:],
            "snapshot_dir": SNAPSHOT_DIR,
            "vision_stats": self.vision_stats,
        }
        path = os.path.join(TELEMETRY_DIR, "%s.json" % self.name)
        try:
            with open(path + ".tmp", "w") as f:
                json.dump(data, f)
            os.replace(path + ".tmp", path)
        except OSError:
            pass      # dashboard file busy - try again next time

    # ----------------------------------------------------------------- run
    def run(self):
        self.robot.step(self.dt_ms)           # first step so sensors have data
        self.sense()
        self.stuck_ref = (self.x, self.y, self.now())
        self.log("START", "InteliPatrol online at (%.1f, %.1f), battery %.0f%%, sector: %s, dock: %s" % (
            self.x, self.y, self.battery, " -> ".join(L.CHECKPOINTS[k][0] for k in self.route), self.dock[0]))
        self.log("INFO", "telemetry -> %s" % TELEMETRY_DIR)
        self.log("INFO", "alerts + snapshots -> %s" % OUTPUT_DIR)
        if self.detector is None:
            self.log("VISION", "disabled")
        elif self.detector.ok:
            self.log("VISION", "%s ready (%s, enhancement %s), camera %dx%d, campus clock %s (%s)" % (
                self.model_name, "campus-trained" if self.detector.custom else "COCO-pretrained",
                "off" if self.detector.custom else ENHANCE_MODE,
                self.cam["width"], self.cam["height"], self.clock.text(self.now()),
                "night" if self.clock.is_night(self.now()) else "day"))
        else:
            self.log("VISION", "YOLO not available (%s) - patrolling without AI vision"
                     % self.detector.error)

        handlers = {"PATROL": self.do_navigate, "RETURN_TO_BASE": self.do_navigate,
                    "SCAN": self.do_scan, "AVOID": self.do_avoid,
                    "CHARGING": self.do_charging, "INVESTIGATE": self.do_investigate,
                    "ALERT": self.do_alert}

        while self.robot.step(self.dt_ms) != -1:
            self.sense()
            self.update_battery()
            self.coordinate()
            self.perceive()
            self.check_unknown_objects()
            if self.battery <= BATTERY_LOW and self.state in ("PATROL", "SCAN"):
                self.log("BATTERY", "battery low (%.0f%%) - returning to base" % self.battery)
                self.path = None
                self.set_state("RETURN_TO_BASE")
            v, w = handlers.get(self.state, self.do_navigate)()
            self.drive(v, w)
            self.write_telemetry()


if __name__ == "__main__":
    PatrolBot().run()
