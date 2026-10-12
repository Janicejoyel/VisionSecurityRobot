"""
planner.py - A* (A-star) path planning on the cost map.

A* explores the grid outward from the start, always expanding the cell with
the lowest  f = g + h, where
  g = cost of the path found so far from the start
  h = estimated cost still remaining to the goal (octile distance)
Because h never over-estimates, A* is guaranteed to find the cheapest path.
Moves are 8-directional; diagonal moves may not cut the corner of a blocked
cell.
"""
import heapq
import math

SQRT2 = math.sqrt(2.0)
MOVES = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
         (1, 1, SQRT2), (1, -1, SQRT2), (-1, 1, SQRT2), (-1, -1, SQRT2)]


def _octile(a, b):
    dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
    return (dx + dy) + (SQRT2 - 2) * min(dx, dy)


def astar(cm, start_xy, goal_xy):
    """Plan from start to goal (world coordinates).
    Returns (list of (x, y) waypoints, number of cells expanded) or (None, n)."""
    start = cm.nearest_free(*cm.world_to_cell(*start_xy))
    goal = cm.nearest_free(*cm.world_to_cell(*goal_xy))
    cost, blocked, n = cm.cost, cm.blocked, cm.n

    g = {start: 0.0}
    came_from = {}
    heap = [(_octile(start, goal), 0.0, start)]
    closed = set()
    expanded = 0

    while heap:
        _, g_cur, cur = heapq.heappop(heap)
        if cur in closed:
            continue
        closed.add(cur)
        expanded += 1
        if cur == goal:
            break
        ci, cj = cur
        for di, dj, step in MOVES:
            ni, nj = ci + di, cj + dj
            if not (0 <= ni < n and 0 <= nj < n) or blocked[ni, nj]:
                continue
            if di and dj and (blocked[ci + di, cj] or blocked[ci, cj + dj]):
                continue                      # no corner cutting
            ng = g_cur + step * 0.5 * (cost[ci, cj] + cost[ni, nj])
            if ng < g.get((ni, nj), math.inf):
                g[(ni, nj)] = ng
                came_from[(ni, nj)] = cur
                heapq.heappush(heap, (ng + _octile((ni, nj), goal), ng, (ni, nj)))
    else:
        return None, expanded

    if goal not in g:
        return None, expanded

    # walk back from goal to start
    cells = [goal]
    while cells[-1] != start:
        cells.append(came_from[cells[-1]])
    cells.reverse()

    # keep every 2nd cell (smoother, fewer waypoints) + the exact goal
    pts = [cm.cell_to_world(i, j) for i, j in cells[::2]]
    if not cm.is_blocked(*cm.world_to_cell(*goal_xy)):
        pts.append(tuple(goal_xy))
    else:
        pts.append(cm.cell_to_world(*goal))
    return pts, expanded


def path_length(path):
    return sum(math.dist(path[k], path[k + 1]) for k in range(len(path) - 1))
