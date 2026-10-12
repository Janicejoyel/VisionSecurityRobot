"""
campus_layout.py - Single source of truth for the InteliPatrol campus.

Units: metres.  Frame (Webots ENU): x = east, y = north, z = up.

This file is used by:
  * tools/build_world.py  -> generates worlds/campus.wbt (the 3D world)
  * the robot controller  -> builds the robot's map for A* path planning

Change the campus here, then re-run build_world.py, and the world and the
robot's map stay in sync.
"""
import math

WORLD_HALF = 20.0          # campus is 40 m x 40 m, from -20 to +20
WALL_THICKNESS = 0.3
WALL_HEIGHT = 1.2

# (name, centre (x, y), size (sx, sy), height, colour (r, g, b))
BUILDINGS = [
    ("Main Block",       (0.0, 10.0),   (12.0, 6.0), 8.0,  (0.85, 0.78, 0.65)),
    ("Library",          (-12.0, -8.0), (7.0, 7.0),  6.0,  (0.70, 0.45, 0.35)),
    ("Lab Block",        (12.0, -8.0),  (8.0, 6.0),  7.0,  (0.60, 0.65, 0.75)),
    ("Hostel",           (-13.0, 12.0), (6.0, 8.0),  10.0, (0.80, 0.70, 0.50)),
    ("Canteen",          (13.0, 11.0),  (6.0, 5.0),  4.0,  (0.90, 0.60, 0.30)),
    ("Charging Station", (0.0, -17.8),  (1.6, 0.5),  1.2,  (0.20, 0.20, 0.25)),
    ("Charging Station B", (-4.0, -17.8), (1.6, 0.5), 1.2,  (0.20, 0.20, 0.25)),
]

TREE_TRUNK_RADIUS = 0.25
TREES = [(-5, -13), (5, -13), (-17, -17), (17, -17), (-17, 3), (17, 3),
         (-3, 17), (3, 17), (-5, 1), (5, 1)]

BENCH_HEIGHT = 0.45
# (centre (x, y), size (sx, sy))
BENCHES = [((-2.5, -12.0), (1.6, 0.5)),
           ((2.5, -12.0), (1.6, 0.5)),
           ((8.0, 3.0), (0.5, 1.6)),
           ((-16.0, -1.0), (1.6, 0.5))]

# Obstacles that exist in the world but are NOT in the robot's map.
# The robot must discover them with its LiDAR and re-plan around them.
UNMAPPED_OBSTACLES = [
    ("Construction Barricade", (6.0, -4.5), (0.4, 3.5), 1.0, (1.0, 0.45, 0.0)),
    ("Maintenance Cart",       (-9.5, 3.0), (1.5, 1.0), 1.1, (0.95, 0.85, 0.1)),
]

# Visual-only ground markings: (name, centre, size, colour)
ROADS = [
    ("road_main_avenue", (0.0, -4.5), (3.0, 23.0), (0.35, 0.35, 0.37)),
    ("road_east_west",   (0.0, -3.0), (36.0, 3.0), (0.35, 0.35, 0.37)),
    ("road_east",        (8.0, 4.0),  (2.5, 14.0), (0.35, 0.35, 0.37)),
    ("road_west",        (-8.0, 4.5), (2.5, 15.0), (0.35, 0.35, 0.37)),
]
SPORTS_GROUND = ((0.0, -9.0), (7.0, 4.0), (0.72, 0.38, 0.28))

# Patrol checkpoints, visited in this order
CHECKPOINTS = [
    ("Sports Ground",       (0.0, -8.0)),
    ("Lab Block Entrance",  (12.0, -3.0)),
    ("Canteen",             (8.0, 11.0)),
    ("Main Block Entrance", (0.0, 5.0)),
    ("Hostel Gate",         (-8.0, 12.0)),
    ("Library Entrance",    (-12.0, -3.0)),
]

DOCKS = [("Charging Dock A", (0.0, -16.0)),
         ("Charging Dock B", (-4.0, -16.0))]
DOCK = DOCKS[0]
ROBOT_START = (0.0, -16.0, math.pi / 2)   # x, y, heading (facing north)

# ---------------------------------------------------------------- Layer 3
# The robot fleet. Each robot patrols its own sector (checkpoint indices) and has
# its own dock. Priority 1 = highest (gets right of way when robots meet).
# start battery < 100 for robot 2 so the demo shows it charging (and robot 1
# taking over its sector) early in the run.
FLEET = [
    {"name": "patrolbot_1", "route": [0, 1, 2], "dock": 0, "battery": 100,
     "start": (0.0, -16.0, math.pi / 2), "color": (0.12, 0.32, 0.78)},
    {"name": "patrolbot_2", "route": [3, 4, 5], "dock": 1, "battery": 55,
     "start": (-4.0, -16.0, math.pi / 2), "color": (0.78, 0.18, 0.55)},
]


def static_boxes():
    """All mapped rectangular obstacles as (cx, cy, sx, sy)."""
    boxes = [(c[0], c[1], s[0], s[1]) for _, c, s, _, _ in BUILDINGS]
    boxes += [(c[0], c[1], s[0], s[1]) for c, s in BENCHES]
    h, t = WORLD_HALF, WALL_THICKNESS
    boxes += [(0, h, 2 * h, t), (0, -h, 2 * h, t), (h, 0, t, 2 * h), (-h, 0, t, 2 * h)]
    return boxes


def static_circles():
    """All mapped round obstacles as (cx, cy, r)."""
    circles = [(x, y, TREE_TRUNK_RADIUS) for x, y in TREES]
    circles += [(x, y, 0.1) for x, y in globals().get("STREET_LAMPS", [])]   # lamp poles are on the map
    return circles


# ---------------------------------------------------------------- FOCV: night mode
# NIGHT_MODE makes the 3D world genuinely dark (moonlight + street lamps) and adds
# camera sensor noise, so the vision system has to cope with low-light images.
NIGHT_MODE = True
CAMERA_NOISE = 0.04 if NIGHT_MODE else 0.0
# street lamps: (x, y) - a pole with a warm light 4 m above the ground
STREET_LAMPS = [(1.9, -12.0), (-1.9, -4.5), (1.9, 3.0), (-6.4, 10.0), (9.6, 8.0), (-9.6, -1.2)]

# ---------------------------------------------------------------- Layer 2
# Simulated campus clock: the demo starts late at night (night patrol).
CLOCK_START_HOUR = 22.5          # 22:30
NIGHT_START, NIGHT_END = 21, 6   # 21:00 - 06:00 counts as night

# Security zones: (name, (xmin, xmax, ymin, ymax), restricted_at_night)
ZONES = [
    ("Hostel Zone",      (-19.7, -6.5, 7.0, 19.7),  True),
    ("Lab Block Zone",   (7.0, 17.5, -12.5, -1.0),  True),
    ("Library Zone",     (-17.5, -6.5, -12.5, 0.5), False),
    ("Canteen Zone",     (8.5, 19.7, 7.5, 19.7),    False),
    ("Main Block Zone",  (-6.5, 7.0, 6.5, 19.7),    False),
    ("Sports Ground",    (-5.0, 5.0, -12.0, -6.0),  False),
]

# People walking around campus (Webots Pedestrian model).
# (name, list of (x, y) waypoints walked in a loop, speed m/s, shirt colour)
PEDESTRIANS = [
    ("hostel_intruder",  [(-7.5, 17.0), (-9.5, 18.2)],                0.35, (0.10, 0.10, 0.12)),
    ("night_walker",     [(-3.5, -10.5), (3.5, -10.5)],               0.6,  (0.20, 0.45, 0.80)),
    ("crowd_member_1",   [(9.5, 15.8), (11.5, 15.8), (10.5, 17.6)],   0.3,  (0.80, 0.20, 0.20)),
    ("crowd_member_2",   [(11.5, 15.8), (10.5, 17.6), (9.5, 15.8)],   0.3,  (0.85, 0.75, 0.15)),
    ("crowd_member_3",   [(10.5, 17.6), (9.5, 15.8), (11.5, 15.8)],   0.3,  (0.25, 0.65, 0.30)),
]

# An unattended suitcase left near the library bench (not on the robot's map).
SUITCASE = ("unattended_suitcase", (-14.0, -1.4))


# ---------------------------------------------------------------- FOCV: dataset world
# Used only by worlds/dataset_capture.wbt (automatic labelled-dataset generation).
DATASET_PEDESTRIANS = [
    ("ds_ped_1", [(-15, -3), (15, -3)],                0.9, (0.80, 0.20, 0.20)),
    ("ds_ped_2", [(0, -14), (0, 4)],                   0.8, (0.20, 0.45, 0.80)),
    ("ds_ped_3", [(8, -1), (8, 14), (15, 16)],         0.7, (0.95, 0.85, 0.20)),
    ("ds_ped_4", [(-8, -1), (-8, 17), (-15, 18.5)],    0.6, (0.10, 0.10, 0.12)),
    ("ds_ped_5", [(-3, -10.5), (3, -10.5)],            0.5, (0.90, 0.90, 0.90)),
    ("ds_ped_6", [(-17, -17), (-17, 1), (-12, 1)],     0.7, (0.25, 0.65, 0.30)),
    ("ds_ped_7", [(17, -17), (17, 1), (11, 1)],        0.8, (0.55, 0.25, 0.65)),
    ("ds_ped_8", [(10.5, 17.5), (12, 16), (9.5, 15.8)], 0.3, (0.95, 0.55, 0.15)),
]
# (size sx, sy, sz, colour) - the data camera moves them to random places
DATASET_SUITCASES = [
    ((0.35, 0.20, 0.50), (0.08, 0.08, 0.09)),
    ((0.35, 0.20, 0.50), (0.10, 0.20, 0.55)),
    ((0.45, 0.25, 0.62), (0.55, 0.08, 0.10)),
    ((0.45, 0.25, 0.62), (0.45, 0.45, 0.48)),
    ((0.55, 0.30, 0.75), (0.10, 0.35, 0.15)),
    ((0.55, 0.30, 0.75), (0.40, 0.25, 0.12)),
]
