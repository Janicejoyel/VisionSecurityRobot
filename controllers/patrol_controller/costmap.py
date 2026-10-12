"""
costmap.py - The robot's internal map of the campus (an occupancy grid).

The campus is divided into small square cells (default 0.4 m).  Each cell is
either free or occupied.  From that we build a *cost map*:

  * cells inside or too close to an obstacle (inflation radius) -> blocked
  * cells near an obstacle get an extra cost -> A* prefers the middle of paths
  * everything else costs 1

Two layers of obstacles are combined:
  * static  - known buildings, trees, benches, walls (from campus_layout.py)
  * dynamic - things the LiDAR sees that are not on the map (barricades,
              people, other robots).  Each cell remembers when it was first
              and last seen:
                - seen for a while (>= `persist_after` s) -> STATIC obstacle
                  (e.g. a barricade), remembered for `decay` seconds
                - seen only briefly -> MOVING obstacle (e.g. a walking person),
                  forgotten after `transient_decay` seconds so people do not
                  leave a "trail" that blocks the road
"""
import math
from collections import deque

import numpy as np

import campus_layout as L


def _shift(a, di, dj):
    """Shift a 2D array by (di, dj) without wrapping around the edges."""
    n0, n1 = a.shape
    out = np.zeros_like(a)
    out[max(0, di):n0 - max(0, -di), max(0, dj):n1 - max(0, -dj)] = \
        a[max(0, -di):n0 - max(0, di), max(0, -dj):n1 - max(0, dj)]
    return out


class CostMap:
    def __init__(self, resolution=0.4, inflation=0.65, proximity=1.6,
                 proximity_penalty=4.0, decay=90.0, transient_decay=1.5,
                 persist_after=3.0):
        self.res = resolution
        self.half = L.WORLD_HALF
        self.n = int(round(2 * self.half / resolution))
        self.inflation = inflation
        self.proximity = proximity
        self.penalty = proximity_penalty
        self.decay = decay
        self.transient_decay = transient_decay
        self.persist_after = persist_after

        centres = -self.half + (np.arange(self.n) + 0.5) * resolution
        # X[i, j] is the x of cell (i, j); Y[i, j] is its y
        self.X, self.Y = np.meshgrid(centres, centres, indexing="ij")

        # neighbour offsets (in cells) up to the largest radius we care about
        r_cells = (inflation + proximity) / resolution
        R = int(math.ceil(r_cells))
        self._offsets = [(di, dj, math.hypot(di, dj) * resolution)
                         for di in range(-R, R + 1) for dj in range(-R, R + 1)
                         if math.hypot(di, dj) <= r_cells]

        self.static = self._rasterise_static()
        self.dynamic = {}          # (i, j) -> [first seen, last seen, times seen]
        self.obs_dt = 0.032        # seconds between LiDAR scans (set by the controller)
        # distance (m) from every cell to the nearest KNOWN (mapped) obstacle
        self.static_dist = self._distance_field(self.static)
        self.version = 0           # increases every time the map changes
        self.rebuild()

    # ------------------------------------------------------------------ build
    def _rasterise_static(self):
        occ = np.zeros((self.n, self.n), dtype=bool)
        pad = self.res / 2
        for cx, cy, sx, sy in L.static_boxes():
            occ |= (np.abs(self.X - cx) <= sx / 2 + pad) & (np.abs(self.Y - cy) <= sy / 2 + pad)
        for cx, cy, r in L.static_circles():
            occ |= np.hypot(self.X - cx, self.Y - cy) <= r + pad
        return occ

    def _distance_field(self, occ):
        """Distance (m) from every cell to the nearest occupied cell (capped)."""
        dist = np.where(occ, 0.0, np.inf)
        for di, dj, d in self._offsets:
            if d == 0:
                continue
            near = _shift(occ, di, dj)
            dist = np.where(near & (d < dist), d, dist)
        return dist

    def rebuild(self):
        occ = self.static.copy()
        for (i, j) in self.dynamic:
            occ[i, j] = True
        dist = self._distance_field(occ)
        self.dist = dist
        self.blocked = dist <= self.inflation
        extra = np.clip(1.0 - (dist - self.inflation) / self.proximity, 0.0, 1.0)
        self.cost = 1.0 + self.penalty * extra
        self.cost[self.blocked] = np.inf
        self.version += 1

    # --------------------------------------------------------------- helpers
    def world_to_cell(self, x, y):
        i = int((x + self.half) / self.res)
        j = int((y + self.half) / self.res)
        return min(max(i, 0), self.n - 1), min(max(j, 0), self.n - 1)

    def cell_to_world(self, i, j):
        return (-self.half + (i + 0.5) * self.res, -self.half + (j + 0.5) * self.res)

    def in_grid(self, x, y):
        return -self.half <= x < self.half and -self.half <= y < self.half

    def is_blocked(self, i, j):
        return bool(self.blocked[i, j])

    def nearest_free(self, i, j):
        """Closest non-blocked cell (breadth-first search)."""
        if not self.blocked[i, j]:
            return i, j
        seen = {(i, j)}
        q = deque([(i, j)])
        while q:
            a, b = q.popleft()
            for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                na, nb = a + da, b + db
                if 0 <= na < self.n and 0 <= nb < self.n and (na, nb) not in seen:
                    if not self.blocked[na, nb]:
                        return na, nb
                    seen.add((na, nb))
                    q.append((na, nb))
        return i, j

    # --------------------------------------------------------- dynamic layer
    def mark_obstacles(self, points, now):
        """Add LiDAR hit points (world x, y) to the dynamic layer.
        Returns True if a new, previously unknown cell was added."""
        added = False
        cells = set()
        for x, y in points:
            if self.in_grid(x, y):
                c = self.world_to_cell(x, y)
                if not self.static[c]:
                    cells.add(c)
        for c in cells:                       # each cell counted once per scan
            if c not in self.dynamic:
                added = True
                self.dynamic[c] = [now, now, 1]
            else:
                self.dynamic[c][1] = now
                self.dynamic[c][2] += 1
        return added

    def forget_old(self, now):
        """Forget dynamic obstacles that have not been seen for a while."""
        old = [c for c, v in self.dynamic.items()
               if now - v[1] > (self.decay if self.is_persistent(v) else self.transient_decay)]
        for c in old:
            del self.dynamic[c]
        return bool(old)

    def is_persistent(self, v):
        """A cell is a STATIC obstacle if it was seen for a while AND seen in most
        scans during that time (a person pacing up and down only crosses a cell now
        and then, so they stay 'moving')."""
        t0, t1, n = v
        span = t1 - t0
        return span >= self.persist_after and n * self.obs_dt / span >= 0.35

    def small_unknown_objects(self, max_size=0.9, min_clearance=0.8):
        """Map change detection: groups of persistent, unmapped cells that are small
        (<= max_size m) and not touching anything on the map.  These are candidate
        'left objects' (bags, parcels...).  Returns a list of (x, y, size_m)."""
        cells = {c for c, v in self.dynamic.items() if self.is_persistent(v)
                 and self.static_dist[c] >= min_clearance}
        found, seen = [], set()
        for c in cells:
            if c in seen:
                continue
            comp, stack = [], [c]
            seen.add(c)
            while stack:                               # flood fill (8-neighbours)
                a, b = stack.pop()
                comp.append((a, b))
                for da in (-1, 0, 1):
                    for db in (-1, 0, 1):
                        nb = (a + da, b + db)
                        if nb in cells and nb not in seen:
                            seen.add(nb)
                            stack.append(nb)
            ii = [a for a, _ in comp]
            jj = [b for _, b in comp]
            size = (max(max(ii) - min(ii), max(jj) - min(jj)) + 1) * self.res
            # ignore it if it is part of something bigger (e.g. the end of a barricade)
            touching = any(nbr in self.dynamic and nbr not in cells
                           for a, b in comp for nbr in ((a + 1, b), (a - 1, b), (a, b + 1), (a, b - 1)))
            if size <= max_size and not touching:
                xs = [self.cell_to_world(a, b) for a, b in comp]
                found.append((sum(x for x, _ in xs) / len(xs), sum(y for _, y in xs) / len(xs), size))
        return found

    def obstacle_counts(self):
        static = sum(1 for v in self.dynamic.values() if self.is_persistent(v))
        return static, len(self.dynamic) - static

    def path_blocked(self, path, start_index=0):
        for x, y in path[start_index:]:
            if self.blocked[self.world_to_cell(x, y)]:
                return True
        return False
