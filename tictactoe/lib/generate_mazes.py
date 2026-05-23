"""
Generate maze puzzle data for cylinder, torus, and mobius topological spaces.

Output: maze_data.json (next to this script)

15 stages = 3 spaces (cylinder/torus/mobius) x 5 difficulty levels.
Each stage stores walls grid + start/end + min wrap requirement.

Algorithm: Randomized DFS spanning tree (including wrap edges as candidates),
then BFS verification of three conditions.
"""
from __future__ import annotations

import json
import random
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple

random.seed(2026)

# -----------------------------------------------------------------------------
# Stage definitions
# -----------------------------------------------------------------------------
STAGE_PLAN: List[Dict] = [
    # cylinder 1-5
    {"key": "cylinder_1", "space": "cylinder", "size": 7,  "min_wraps": 1},
    {"key": "cylinder_2", "space": "cylinder", "size": 9,  "min_wraps": 1},
    {"key": "cylinder_3", "space": "cylinder", "size": 11, "min_wraps": 2},
    {"key": "cylinder_4", "space": "cylinder", "size": 13, "min_wraps": 2},
    {"key": "cylinder_5", "space": "cylinder", "size": 15, "min_wraps": 3},
    # torus 1-5
    {"key": "torus_1", "space": "torus", "size": 7,  "min_wraps": 1},
    {"key": "torus_2", "space": "torus", "size": 9,  "min_wraps": 2},
    {"key": "torus_3", "space": "torus", "size": 11, "min_wraps": 2},
    {"key": "torus_4", "space": "torus", "size": 13, "min_wraps": 3},
    {"key": "torus_5", "space": "torus", "size": 15, "min_wraps": 4},
    # mobius 1-5
    {"key": "mobius_1", "space": "mobius", "size": 7,  "min_wraps": 1},
    {"key": "mobius_2", "space": "mobius", "size": 9,  "min_wraps": 1},
    {"key": "mobius_3", "space": "mobius", "size": 11, "min_wraps": 2},
    {"key": "mobius_4", "space": "mobius", "size": 13, "min_wraps": 3},
    {"key": "mobius_5", "space": "mobius", "size": 15, "min_wraps": 3},
]

# Wall encoding: walls[r][c] = [right_wall, down_wall], 1=wall, 0=open
RIGHT, DOWN = 0, 1


# -----------------------------------------------------------------------------
# Neighbor / wall accessor
# -----------------------------------------------------------------------------
def neighbors_with_edge(r: int, c: int, rows: int, cols: int, space: str):
    """Yield (nr, nc, wall_owner_r, wall_owner_c, wall_dir, is_wrap)
    where wall_owner identifies which cell stores the wall flag, and
    wall_dir is RIGHT or DOWN.
    """
    # right
    if c + 1 < cols:
        yield (r, c + 1, r, c, RIGHT, False)
    else:
        # right edge of grid
        if space == "cylinder":
            yield (r, 0, r, c, RIGHT, True)
        elif space == "torus":
            yield (r, 0, r, c, RIGHT, True)
        elif space == "mobius":
            yield (rows - 1 - r, 0, r, c, RIGHT, True)
    # left
    if c - 1 >= 0:
        yield (r, c - 1, r, c - 1, RIGHT, False)
    else:
        # left edge
        if space == "cylinder":
            yield (r, cols - 1, r, cols - 1, RIGHT, True)
        elif space == "torus":
            yield (r, cols - 1, r, cols - 1, RIGHT, True)
        elif space == "mobius":
            # cell (r, 0).left = (rows-1-r, cols-1)
            yield (rows - 1 - r, cols - 1, rows - 1 - r, cols - 1, RIGHT, True)
    # down
    if r + 1 < rows:
        yield (r + 1, c, r, c, DOWN, False)
    else:
        if space == "torus":
            yield (0, c, r, c, DOWN, True)
        # cylinder, mobius: top-bottom = wall (no edge)
    # up
    if r - 1 >= 0:
        yield (r - 1, c, r - 1, c, DOWN, False)
    else:
        if space == "torus":
            yield (rows - 1, c, rows - 1, c, DOWN, True)


# -----------------------------------------------------------------------------
# Maze generation: randomized DFS spanning tree
# -----------------------------------------------------------------------------
def generate_maze(rows: int, cols: int, space: str) -> List[List[List[int]]]:
    """Returns walls grid (all walls present initially, then carve)."""
    walls = [[[1, 1] for _ in range(cols)] for _ in range(rows)]

    visited = [[False] * cols for _ in range(rows)]
    start = (0, 0)
    stack = [start]
    visited[0][0] = True

    while stack:
        r, c = stack[-1]
        # collect unvisited neighbors
        nbrs = []
        for nr, nc, wr, wc, wd, _is_wrap in neighbors_with_edge(r, c, rows, cols, space):
            if not visited[nr][nc]:
                nbrs.append((nr, nc, wr, wc, wd))
        if not nbrs:
            stack.pop()
            continue
        nr, nc, wr, wc, wd = random.choice(nbrs)
        # carve wall
        walls[wr][wc][wd] = 0
        visited[nr][nc] = True
        stack.append((nr, nc))

    return walls


# -----------------------------------------------------------------------------
# BFS with wrap-aware adjacency (also counts wraps used)
# -----------------------------------------------------------------------------
def bfs_with_wraps(walls, rows, cols, space, start, end, allow_wrap=True):
    """BFS from start. Returns (reachable, distance_to_end, wraps_in_shortest_path).
    If allow_wrap=False, treats all wrap edges as walls (plain mode).
    Tracks wraps along the discovered shortest path via parent pointers.
    """
    INF = float("inf")
    dist = [[INF] * cols for _ in range(rows)]
    wraps = [[0] * cols for _ in range(rows)]
    parent = [[None] * cols for _ in range(rows)]
    dist[start[0]][start[1]] = 0
    q = deque([start])

    while q:
        r, c = q.popleft()
        for nr, nc, wr, wc, wd, is_wrap in neighbors_with_edge(r, c, rows, cols, space):
            if not allow_wrap and is_wrap:
                continue
            if walls[wr][wc][wd] == 1:
                continue
            if dist[nr][nc] == INF:
                dist[nr][nc] = dist[r][c] + 1
                wraps[nr][nc] = wraps[r][c] + (1 if is_wrap else 0)
                parent[nr][nc] = (r, c, is_wrap)
                q.append((nr, nc))

    er, ec = end
    if dist[er][ec] == INF:
        return False, None, None

    # Count wraps on the BFS shortest path
    wrap_count = 0
    cur = (er, ec)
    while parent[cur[0]][cur[1]] is not None:
        pr, pc, is_wrap = parent[cur[0]][cur[1]]
        if is_wrap:
            wrap_count += 1
        cur = (pr, pc)

    return True, dist[er][ec], wrap_count


# -----------------------------------------------------------------------------
# Verify maze satisfies all three conditions
# -----------------------------------------------------------------------------
def verify_maze(walls, rows, cols, space, start, end, min_wraps):
    # Condition A: reachable in space mode
    ok_a, dist_a, wraps_a = bfs_with_wraps(walls, rows, cols, space, start, end, allow_wrap=True)
    if not ok_a:
        return False, None
    # Condition B: NOT reachable in plain mode (no wraps)
    ok_b, _, _ = bfs_with_wraps(walls, rows, cols, space, start, end, allow_wrap=False)
    if ok_b:
        return False, None
    # Condition C: shortest path wraps >= min_wraps
    if wraps_a < min_wraps:
        return False, None
    return True, {"solutionLength": dist_a, "solutionWraps": wraps_a, "plainReachable": False}


# -----------------------------------------------------------------------------
# Try to find a valid maze for a stage
# -----------------------------------------------------------------------------
def make_stage(key, space, size, min_wraps):
    rows = cols = size
    grid_size = size
    while grid_size <= size + 20:
        for attempt in range(1000):
            walls = generate_maze(grid_size, grid_size, space)
            mid = grid_size // 2
            start = (mid, mid)
            end = (grid_size - 1, grid_size - 1)
            ok, info = verify_maze(walls, grid_size, grid_size, space, start, end, min_wraps)
            if ok:
                return {
                    "rows": grid_size,
                    "cols": grid_size,
                    "walls": walls,
                    "start": [mid, mid],
                    "end": [grid_size - 1, grid_size - 1],
                    "space": space,
                    "requiredWraps": min_wraps,
                    "solutionLength": info["solutionLength"],
                    "solutionWraps": info["solutionWraps"],
                    "plainReachable": info["plainReachable"],
                    "grid_escalation": grid_size - size,
                }
        # escalate +2
        grid_size += 2
    raise RuntimeError(f"Failed to generate {key} even with escalation")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    data = {}
    rows_report = []
    pass_count = 0

    for plan in STAGE_PLAN:
        key = plan["key"]
        stage = make_stage(key, plan["space"], plan["size"], plan["min_wraps"])

        # Re-verify independently
        walls = stage["walls"]
        rows = stage["rows"]; cols = stage["cols"]
        ok, info = verify_maze(walls, rows, cols, plan["space"],
                               tuple(stage["start"]), tuple(stage["end"]), plan["min_wraps"])
        status = "PASS" if ok else "FAIL"
        if ok:
            pass_count += 1
        data[key] = {
            "rows": stage["rows"],
            "cols": stage["cols"],
            "walls": stage["walls"],
            "start": stage["start"],
            "end": stage["end"],
            "space": stage["space"],
            "requiredWraps": stage["requiredWraps"],
            "solutionLength": stage["solutionLength"],
            "solutionWraps": stage["solutionWraps"],
            "plainReachable": stage["plainReachable"],
        }
        rows_report.append({
            "key": key,
            "space": plan["space"],
            "size": stage["rows"],
            "esc": stage["grid_escalation"],
            "req_wraps": plan["min_wraps"],
            "sol_len": stage["solutionLength"],
            "sol_wraps": stage["solutionWraps"],
            "status": status,
        })

    # Write JSON
    out_path = Path(__file__).parent / "maze_data.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # Report table
    print()
    print(f"{'KEY':<14} {'SPACE':<10} {'SIZE':<6} {'ESC':<5} {'REQ':<5} {'LEN':<5} {'WRAPS':<6} {'STATUS'}")
    print("-" * 70)
    for r in rows_report:
        print(f"{r['key']:<14} {r['space']:<10} {r['size']:<6} {r['esc']:<5} {r['req_wraps']:<5} "
              f"{r['sol_len']:<5} {r['sol_wraps']:<6} {r['status']}")
    print("-" * 70)
    print(f"TOTAL: {pass_count}/{len(STAGE_PLAN)} PASS")
    print(f"Output: {out_path}")
    print(f"Keys in file: {len(data)}")


if __name__ == "__main__":
    main()
