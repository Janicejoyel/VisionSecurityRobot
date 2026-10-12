"""
threat.py - Decides how serious a detection is (threat scoring).

Every detection gets a score from 0 to 100 built from simple, explainable
rules (base score + context bonuses).  The score picks the response:

    score >= 70  HIGH    -> robot goes to INVESTIGATE, then raises an ALERT
    40 .. 69     MEDIUM  -> snapshot + logged alert, robot keeps patrolling
    < 40         LOW     -> just logged

Rules
  person ............................ 20
    + night time (21:00-06:00) ...... +35
    + inside a restricted zone ...... +25   (restricted zones only count at night)
  crowd (3+ people in one frame) .... 50  (+35 at night)
  bag / suitcase / backpack ......... 40
    + nobody within 3 m (unattended)  +35
  unknown object (LiDAR map change) . 45
    + night time .................... +25
"""
import math

import campus_layout as L

HIGH, MEDIUM = 70, 40
BAG_LABELS = {"backpack", "handbag", "suitcase"}
CROWD_SIZE = 3
UNATTENDED_RADIUS = 3.0
SAME_EVENT_RADIUS = 6.0      # m   - don't alert twice for the same thing...
SAME_EVENT_WINDOW = 120.0    # s   - ...within this many seconds


class CampusClock:
    """Simulated time of day: 1 simulated second = 1 campus second."""

    def __init__(self, start_hour=L.CLOCK_START_HOUR):
        self.start = start_hour * 3600.0

    def seconds(self, sim_time):
        return (self.start + sim_time) % 86400.0

    def text(self, sim_time):
        s = int(self.seconds(sim_time))
        return "%02d:%02d" % (s // 3600, (s % 3600) // 60)

    def is_night(self, sim_time):
        h = self.seconds(sim_time) / 3600.0
        return h >= L.NIGHT_START or h < L.NIGHT_END


def zone_of(x, y):
    for name, (x0, x1, y0, y1), restricted in L.ZONES:
        if x0 <= x <= x1 and y0 <= y <= y1:
            return name, restricted
    return "Open Campus", False


def level_of(score):
    return "HIGH" if score >= HIGH else "MEDIUM" if score >= MEDIUM else "LOW"


class ThreatAssessor:
    def __init__(self, clock):
        self.clock = clock
        self.memory = []          # recent events: (kind, x, y, t)

    def assess(self, located, sim_time):
        """located: detections with a 'pos' {x, y}. Returns a list of threats."""
        night = self.clock.is_night(sim_time)
        people = [d for d in located if d["label"] == "person"]
        bags = [d for d in located if d["label"] in BAG_LABELS]
        threats = []

        if len(people) >= CROWD_SIZE:
            cx = sum(d["pos"]["x"] for d in people) / len(people)
            cy = sum(d["pos"]["y"] for d in people) / len(people)
            reasons = ["crowd of %d people (+50)" % len(people)]
            score = 50
            if night:
                score += 35
                reasons.append("night time (+35)")
            threats.append(self._make("CROWD", score, cx, cy, reasons, people))
        else:
            for d in people:
                x, y = d["pos"]["x"], d["pos"]["y"]
                zone, restricted = zone_of(x, y)
                score, reasons = 20, ["person detected (+20)"]
                if night:
                    score += 35
                    reasons.append("night time (+35)")
                    if restricted:
                        score += 25
                        reasons.append("restricted zone at night (+25)")
                kind = "INTRUDER" if score >= HIGH else "PERSON"
                threats.append(self._make(kind, score, x, y, reasons, [d]))

        for d in bags:
            x, y = d["pos"]["x"], d["pos"]["y"]
            score, reasons = 40, ["%s detected (+40)" % d["label"]]
            near = [p for p in people
                    if math.hypot(p["pos"]["x"] - x, p["pos"]["y"] - y) < UNATTENDED_RADIUS]
            if not near:
                score += 35
                reasons.append("no owner within %.0f m (+35)" % UNATTENDED_RADIUS)
            kind = "UNATTENDED_OBJECT" if not near else "BAG"
            threats.append(self._make(kind, score, x, y, reasons, [d]))

        threats.sort(key=lambda t: -t["score"])
        return threats

    def _make(self, kind, score, x, y, reasons, dets):
        zone, _ = zone_of(x, y)
        return {"kind": kind, "score": min(100, score), "level": level_of(score),
                "x": round(x, 2), "y": round(y, 2), "zone": zone,
                "reasons": reasons, "detections": dets}

    def loitering(self, trk, sim_time):
        """A tracked person who has stayed in one small area for a long time."""
        zone, restricted = zone_of(trk.x, trk.y)
        score = 40
        reasons = ["person #%d stayed within %.1f m for %.0f s (+40)" % (
            trk.id, 2.5, trk.last_seen - trk.first_seen)]
        if self.clock.is_night(sim_time):
            score += 35
            reasons.append("night time (+35)")
            if restricted:
                score += 25
                reasons.append("restricted zone at night (+25)")
        return self._make("LOITERING", score, trk.x, trk.y, reasons, [])

    def unknown_object(self, x, y, size, sim_time):
        """A small object found by LiDAR map change detection."""
        score, reasons = 45, ["unmapped object %.1f m wide appeared (+45)" % size]
        if self.clock.is_night(sim_time):
            score += 25
            reasons.append("night time (+25)")
        return self._make("UNKNOWN_OBJECT", score, x, y, reasons, [])

    # -------------------------------------------------- duplicate filtering
    @staticmethod
    def group(kind):
        if kind in ("PERSON", "INTRUDER", "CROWD", "LOITERING"):
            return "people"
        if kind in ("UNATTENDED_OBJECT", "BAG", "UNKNOWN_OBJECT", "UNIDENTIFIED_OBJECT"):
            return "object"
        return kind

    def is_new(self, threat, sim_time):
        g = self.group(threat["kind"])
        radius = SAME_EVENT_RADIUS if g == "people" else 2.5
        window = SAME_EVENT_WINDOW if g == "people" else 1e9    # an object is reported once
        for kind, x, y, t in self.memory:
            if (self.group(kind) == g and sim_time - t < window and
                    math.hypot(x - threat["x"], y - threat["y"]) < radius):
                if g == "people" and threat["score"] > self._score_of(kind):
                    continue          # an escalation (e.g. person -> crowd) is still news
                return False
        return True

    @staticmethod
    def _score_of(kind):
        return {"PERSON": 55, "INTRUDER": 80, "LOITERING": 99, "CROWD": 85}.get(kind, 0)

    def remember(self, threat, sim_time):
        self.memory.append((threat["kind"], threat["x"], threat["y"], sim_time))
        self.memory = [m for m in self.memory
                       if sim_time - m[3] < SAME_EVENT_WINDOW or self.group(m[0]) == "object"]
