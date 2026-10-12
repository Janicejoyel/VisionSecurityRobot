"""
tracker.py - Multi-object tracking of people in MAP coordinates.

Why not a normal image tracker (SORT / ByteTrack)?  Those follow boxes inside the
image and assume a still camera. Our camera rides on a moving, rotating robot, so a
person "jumps" across the image every time the robot turns. Instead we track in
campus-map coordinates (from camera + LiDAR fusion), which do not move when the
robot turns.

Each step:
  1. PREDICT  - every track moves with its own velocity (constant-velocity model)
  2. COST     - for each (track, detection) pair:
                  motion cost      = distance(prediction, detection) / gate
                  appearance cost  = Bhattacharyya distance between HSV colour
                                     histograms of the clothing (0 = identical)
                  total            = 0.6 * motion + 0.4 * appearance
  3. MATCH    - Hungarian algorithm (minimum total cost assignment, via `lap`)
  4. UPDATE   - alpha-beta filter smooths position and velocity; the colour
                histogram is updated slowly (running average)
  5. MANAGE   - unmatched detections start new tracks; tracks unseen for a while
                are deleted. A person who leaves the view and comes back near the
                same place with the same clothes gets their OLD id back
                (re-identification), because tracks are kept alive while unseen.

Behaviour analysis on top of the tracks:
  * speed      - |velocity| in m/s
  * loitering  - stayed within LOITER_RADIUS for at least LOITER_TIME seconds
"""
import math

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None
try:
    import lap
except ImportError:          # fall back to greedy matching
    lap = None

GATE_BASE = 1.5            # m - max match distance for a track seen just now
GATE_GROWTH = 0.8          # m per second unseen (people may have walked on)
GATE_MAX = 4.0             # m
W_MOTION, W_APPEAR = 0.6, 0.4
MAX_COST = 0.9
ALPHA, BETA = 0.6, 0.3     # alpha-beta filter gains (position, velocity)
MAX_UNSEEN = 30.0          # s - delete a track not seen for this long
CONFIRM_HITS = 2           # detections before a track is "confirmed"
LOITER_RADIUS = 2.5        # m
LOITER_TIME = 20.0         # s
TRAIL_LEN = 40


def colour_histogram(frame, box):
    """Normalised 2D Hue-Saturation histogram of the middle of a bounding box
    (the torso / clothing). HSV separates colour from brightness, so the same
    shirt looks similar under a street lamp and in shadow."""
    if cv2 is None or frame is None:
        return None
    x1, y1, x2, y2 = box
    h, w = y2 - y1, x2 - x1
    crop = frame[max(0, y1 + h // 5): max(0, y2 - h // 4), max(0, x1 + w // 5): max(0, x2 - w // 5)]
    if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
    return hist


def appearance_distance(h1, h2):
    if h1 is None or h2 is None or cv2 is None:
        return 0.5                                     # unknown -> neutral
    return float(cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA))


class Track:
    def __init__(self, tid, det, t):
        self.id = tid
        self.label = det["label"]
        self.x, self.y = det["pos"]["x"], det["pos"]["y"]
        self.vx = self.vy = 0.0
        self.hist = det.get("hist")
        self.first_seen = self.last_seen = self.last_update = t
        self.hits = 1
        self.trail = [(t, self.x, self.y)]
        self.loitering = False
        self.loiter_reported = False
        self.conf = det["conf"]

    def predict(self, t):
        dt = t - self.last_update
        return self.x + self.vx * dt, self.y + self.vy * dt

    def update(self, det, t):
        px, py = self.predict(t)
        mx, my = det["pos"]["x"], det["pos"]["y"]
        dt = max(1e-3, t - self.last_update)
        rx, ry = mx - px, my - py                      # residual (measurement - prediction)
        self.x, self.y = px + ALPHA * rx, py + ALPHA * ry
        if dt < 5.0:                                   # velocity only from close-in-time updates
            self.vx += BETA * rx / dt
            self.vy += BETA * ry / dt
            sp = math.hypot(self.vx, self.vy)
            if sp > 2.5:                               # people don't run faster than this here
                self.vx, self.vy = self.vx * 2.5 / sp, self.vy * 2.5 / sp
        else:
            self.vx = self.vy = 0.0
        if det.get("hist") is not None:
            self.hist = det["hist"] if self.hist is None else 0.8 * self.hist + 0.2 * det["hist"]
        self.last_seen = self.last_update = t
        self.hits += 1
        self.conf = det["conf"]
        self.trail.append((t, self.x, self.y))
        self.trail = self.trail[-TRAIL_LEN:]

    @property
    def speed(self):
        return math.hypot(self.vx, self.vy)

    def check_loitering(self, t):
        """Loitering = seen over >= LOITER_TIME s and never left a LOITER_RADIUS circle."""
        pts = [(x, y) for ts, x, y in self.trail if t - ts <= LOITER_TIME * 3]
        span = self.last_seen - self.first_seen
        if span < LOITER_TIME or len(pts) < 4:
            self.loitering = False
            return False
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        self.loitering = max(math.hypot(x - cx, y - cy) for x, y in pts) <= LOITER_RADIUS
        return self.loitering


class MapTracker:
    def __init__(self, labels=("person",)):
        self.labels = set(labels)
        self.tracks = []
        self.next_id = 1
        self.total_created = 0

    def _gate(self, trk, t):
        return min(GATE_MAX, GATE_BASE + GATE_GROWTH * (t - trk.last_seen))

    def update(self, dets, t):
        """dets: located detections (with 'pos', optional 'hist'). Adds 'track_id'
        to each detection that was matched or started a new track."""
        dets = [d for d in dets if d["label"] in self.labels]
        tracks = self.tracks
        if tracks and dets:
            cost = np.full((len(tracks), len(dets)), 1e3)
            for i, trk in enumerate(tracks):
                px, py = trk.predict(t)
                gate = self._gate(trk, t)
                for j, d in enumerate(dets):
                    dist = math.hypot(d["pos"]["x"] - px, d["pos"]["y"] - py)
                    if t - trk.last_seen > 2.0:   # unseen a while: they may have turned back,
                        dist = min(dist, math.hypot(d["pos"]["x"] - trk.x, d["pos"]["y"] - trk.y))
                    if dist > gate:
                        continue
                    cost[i, j] = W_MOTION * dist / gate + W_APPEAR * appearance_distance(trk.hist, d.get("hist"))
            pairs = self._assign(cost)
        else:
            pairs = []

        matched_t, matched_d = set(), set()
        for i, j in pairs:
            tracks[i].update(dets[j], t)
            dets[j]["track_id"] = tracks[i].id
            matched_t.add(i)
            matched_d.add(j)
        for j, d in enumerate(dets):
            if j not in matched_d:
                trk = Track(self.next_id, d, t)
                self.next_id += 1
                self.total_created += 1
                tracks.append(trk)
                d["track_id"] = trk.id
        self.tracks = [trk for trk in tracks if t - trk.last_seen <= MAX_UNSEEN]
        for trk in self.tracks:
            trk.check_loitering(t)
        return self.tracks

    @staticmethod
    def _assign(cost):
        """Minimum-cost one-to-one assignment (Hungarian / Jonker-Volgenant)."""
        if lap is not None:
            _, x, _ = lap.lapjv(cost, extend_cost=True, cost_limit=MAX_COST)
            return [(i, j) for i, j in enumerate(x) if j >= 0 and cost[i, j] <= MAX_COST]
        pairs, used_i, used_j = [], set(), set()          # greedy fallback
        for i, j in sorted(np.ndindex(cost.shape), key=lambda ij: cost[ij]):
            if cost[i, j] > MAX_COST:
                break
            if i not in used_i and j not in used_j:
                pairs.append((i, j)); used_i.add(i); used_j.add(j)
        return pairs

    def confirmed(self):
        return [trk for trk in self.tracks if trk.hits >= CONFIRM_HITS]

    def summary(self, t):
        return [{"id": trk.id, "label": trk.label, "x": round(trk.x, 2), "y": round(trk.y, 2),
                 "speed": round(trk.speed, 2), "hits": trk.hits,
                 "age": round(trk.last_seen - trk.first_seen, 1),
                 "unseen": round(t - trk.last_seen, 1), "loitering": trk.loitering,
                 "trail": [[round(x, 2), round(y, 2)] for _, x, y in trk.trail[-20:]]}
                for trk in self.confirmed()]
