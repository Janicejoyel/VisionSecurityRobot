"""
autolabel.py - Automatic bounding-box labels from known 3D geometry + depth.

For every object whose 3D position we know (from the Webots supervisor):

  1. PROJECT  - the 8 corners of the object's 3D box are projected into the image
                with the pinhole camera model:
                    u = W/2 - f * (y_c / x_c)        v = H/2 - f * (z_c / x_c)
                where (x_c, y_c, z_c) is the corner in camera coordinates
                (x forward, y left, z up) and f = (W/2) / tan(hfov/2).
                The min/max of the projected corners is a loose 2D box.
  2. DEPTH    - inside that loose box, keep only pixels whose measured depth is
                within the object's depth range. Pixels in front belong to an
                occluder, pixels behind are background.
  3. GROUND   - pixels whose 3D height is ~0 are the ground under the object
                (same depth as the feet / wheels) and are removed.
  4. TIGHTEN  - the remaining pixels give a tight box (1st-99th percentile).
  5. VISIBLE  - if too few pixels remain, the object is hidden (occluded) or too
                small to label -> it is skipped.
"""
import math

import numpy as np


def focal_px(width, hfov):
    return (width / 2.0) / math.tan(hfov / 2.0)


def to_camera(points, cam):
    """World points (N,3) -> camera coords (x fwd, y left, z up). cam = (x, y, z, yaw)."""
    cx, cy, cz, yaw = cam
    d = np.asarray(points, float) - np.array([cx, cy, cz])
    c, s = math.cos(yaw), math.sin(yaw)
    xc = c * d[:, 0] + s * d[:, 1]
    yc = -s * d[:, 0] + c * d[:, 1]
    return np.stack([xc, yc, d[:, 2]], axis=1)


def project(points_cam, width, height, f):
    x = np.maximum(points_cam[:, 0], 1e-6)
    u = width / 2.0 - f * points_cam[:, 1] / x
    v = height / 2.0 - f * points_cam[:, 2] / x
    return u, v


def box_corners(cx, cy, yaw, sx, sy, z0, z1):
    """8 corners of an upright box centred at (cx, cy), rotated by yaw."""
    c, s = math.cos(yaw), math.sin(yaw)
    pts = []
    for dx in (-sx / 2, sx / 2):
        for dy in (-sy / 2, sy / 2):
            for z in (z0, z1):
                pts.append((cx + c * dx - s * dy, cy + s * dx + c * dy, z))
    return np.array(pts)


def planar_depth(depth, kind, width, height, f):
    """Convert a range image to planar depth (distance along the camera axis)."""
    if kind == "planar":
        return depth
    u = np.arange(width) - width / 2.0
    v = np.arange(height) - height / 2.0
    uu, vv = np.meshgrid(u, v)
    return depth / np.sqrt(1.0 + (uu / f) ** 2 + (vv / f) ** 2)


def label_object(obj, cam, depth_planar, width, height, hfov, min_px=10, min_visible=0.25,
                 depth_tol=0.35, ground_z=0.06):
    """obj: dict(cls, x, y, yaw, sx, sy, z0, z1, fill). Returns a label dict or None."""
    f = focal_px(width, hfov)
    corners = box_corners(obj["x"], obj["y"], obj["yaw"], obj["sx"], obj["sy"], obj["z0"], obj["z1"])
    pc = to_camera(corners, cam)
    if np.all(pc[:, 0] < 0.2):
        return None                                       # behind the camera
    pc = pc[pc[:, 0] >= 0.2]
    if len(pc) < 2:
        return None
    u, v = project(pc, width, height, f)
    u0, u1 = int(max(0, math.floor(u.min()))), int(min(width - 1, math.ceil(u.max())))
    v0, v1 = int(max(0, math.floor(v.min()))), int(min(height - 1, math.ceil(v.max())))
    if u1 - u0 < 3 or v1 - v0 < 3:
        return None                                       # outside the image or tiny
    dmin, dmax = pc[:, 0].min() - depth_tol, pc[:, 0].max() + depth_tol
    patch = depth_planar[v0:v1 + 1, u0:u1 + 1]
    mask = np.isfinite(patch) & (patch >= dmin) & (patch <= dmax)
    # remove ground pixels: height of each pixel = cam_z + (H/2 - v)/f * depth
    vv = (height / 2.0 - np.arange(v0, v1 + 1))[:, None]
    z = cam[2] + vv / f * np.where(np.isfinite(patch), patch, 0.0)
    mask &= z > ground_z
    n = int(mask.sum())
    loose_area = (u1 - u0 + 1) * (v1 - v0 + 1)
    visible = n / max(1.0, obj.get("fill", 0.5) * loose_area)
    if n < 15 or visible < min_visible:
        return None                                       # occluded / not enough pixels
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    ys, xs = np.nonzero(mask)
    bu0, bu1 = u0 + np.percentile(xs, 1), u0 + np.percentile(xs, 99)
    bv0, bv1 = v0 + np.percentile(ys, 1), v0 + np.percentile(ys, 99)
    if bv1 - bv0 < min_px or bu1 - bu0 < 3:
        return None                                       # too small to be useful
    dist = float(math.hypot(obj["x"] - cam[0], obj["y"] - cam[1]))
    truncated = u0 == 0 or v0 == 0 or u1 == width - 1 or v1 == height - 1
    return {"cls": obj["cls"], "box": [float(bu0), float(bv0), float(bu1), float(bv1)],
            "visible": round(min(1.0, visible), 2), "dist": round(dist, 2), "truncated": truncated,
            "name": obj.get("name", "")}


def to_yolo(label, width, height):
    x0, y0, x1, y1 = label["box"]
    return "%d %.6f %.6f %.6f %.6f" % (label["cls"], (x0 + x1) / 2 / width, (y0 + y1) / 2 / height,
                                       (x1 - x0) / width, (y1 - y0) / height)
