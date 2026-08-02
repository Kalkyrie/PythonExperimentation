"""Level state: grid, doors, collision and visibility queries."""
import math
from collections import deque

from mapdata import WALL_CHARS, DOOR_CHARS
from settings import DOOR_SPEED, DOOR_OPEN_TIME, DOOR_PASSABLE

EXIT_ID = 10
EXIT_PRESSED_ID = 11


class Door:
    def __init__(self, key=None):
        self.open = 0.0          # 0 closed .. 1 fully open
        self.state = "closed"    # closed / opening / open / closing
        self.timer = 0.0
        self.key = key

    def update(self, dt, blocked):
        if self.state == "opening":
            self.open = min(1.0, self.open + DOOR_SPEED * dt)
            if self.open >= 1.0:
                self.state = "open"
                self.timer = DOOR_OPEN_TIME
        elif self.state == "open":
            self.timer -= dt
            if self.timer <= 0 and not blocked:
                self.state = "closing"
        elif self.state == "closing":
            self.open = max(0.0, self.open - DOOR_SPEED * dt)
            if self.open <= 0.0:
                self.state = "closed"

    @property
    def passable(self):
        return self.open >= DOOR_PASSABLE


class Level:
    """Parsed level: wall grid, doors, and spawn markers."""

    def __init__(self, data):
        self.name = data["name"]
        self.floor_tex = data["floor"]
        self.floor2_tex = data["floor2"]
        self.ceil_tex = data["ceiling"]
        self.player_angle = data["player_angle"]

        rows = data["grid"]
        self.h = len(rows)
        self.w = max(len(r) for r in rows)
        self.grid = [[0] * self.w for _ in range(self.h)]
        self.doors = {}
        self.player_start = (1.5, 1.5)
        self.spawns = []          # (kind, x, y) for entities/pickups
        self.exit_pos = None

        for y, row in enumerate(rows):
            for x, ch in enumerate(row):
                cx, cy = x + 0.5, y + 0.5
                if ch in WALL_CHARS:
                    self.grid[y][x] = WALL_CHARS[ch]
                    if ch in DOOR_CHARS:
                        self.doors[(x, y)] = Door(DOOR_CHARS[ch])
                    if WALL_CHARS[ch] == EXIT_ID:
                        self.exit_pos = (x, y)
                elif ch == "P":
                    self.player_start = (cx, cy)
                elif ch != ".":
                    self.spawns.append((ch, cx, cy))

    # -- queries ----------------------------------------------------------

    def in_bounds(self, x, y):
        return 0 <= x < self.w and 0 <= y < self.h

    def wall_at(self, x, y):
        if not self.in_bounds(x, y):
            return 1
        return self.grid[y][x]

    def is_solid(self, x, y):
        """Blocks movement?"""
        t = self.wall_at(x, y)
        if t == 0:
            return False
        door = self.doors.get((x, y))
        if door is not None:
            return not door.passable
        return True

    def blocks_sight(self, x, y):
        t = self.wall_at(x, y)
        if t == 0:
            return False
        door = self.doors.get((x, y))
        if door is not None:
            return door.open < 0.35
        return True

    def move_circle(self, x, y, dx, dy, radius, extra_blockers=()):
        """Slide a circle through the grid; returns final (x, y)."""
        nx = self._axis_move(x, y, dx, 0, radius, extra_blockers)[0]
        ny = self._axis_move(nx, y, 0, dy, radius, extra_blockers)[1]
        return nx, ny

    def _axis_move(self, x, y, dx, dy, radius, extra_blockers):
        nx, ny = x + dx, y + dy
        for gy in range(int(ny - radius), int(ny + radius) + 1):
            for gx in range(int(nx - radius), int(nx + radius) + 1):
                if not self.is_solid(gx, gy):
                    continue
                # closest point on cell to circle centre
                cx = min(max(nx, gx), gx + 1)
                cy = min(max(ny, gy), gy + 1)
                if (nx - cx) ** 2 + (ny - cy) ** 2 < radius * radius:
                    if dx > 0:
                        nx = gx - radius
                    elif dx < 0:
                        nx = gx + 1 + radius
                    if dy > 0:
                        ny = gy - radius
                    elif dy < 0:
                        ny = gy + 1 + radius
        for (bx, by, br) in extra_blockers:
            d2 = (nx - bx) ** 2 + (ny - by) ** 2
            rr = radius + br
            if d2 < rr * rr and d2 > 1e-9:
                d = math.sqrt(d2)
                nx = bx + (nx - bx) / d * rr
                ny = by + (ny - by) / d * rr
        return nx, ny

    def line_of_sight(self, x0, y0, x1, y1, max_dist=40.0):
        """DDA ray from A to B; True if no sight-blocking cell between."""
        dx, dy = x1 - x0, y1 - y0
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            return True
        if dist > max_dist:
            return False
        dx /= dist
        dy /= dist
        map_x, map_y = int(x0), int(y0)
        delta_x = abs(1 / dx) if dx else 1e30
        delta_y = abs(1 / dy) if dy else 1e30
        step_x = 1 if dx > 0 else -1
        step_y = 1 if dy > 0 else -1
        side_x = ((map_x + 1 - x0) if dx > 0 else (x0 - map_x)) * delta_x
        side_y = ((map_y + 1 - y0) if dy > 0 else (y0 - map_y)) * delta_y
        travelled = 0.0
        while travelled < dist:
            if side_x < side_y:
                travelled = side_x
                side_x += delta_x
                map_x += step_x
            else:
                travelled = side_y
                side_y += delta_y
                map_y += step_y
            if travelled >= dist:
                return True
            if self.blocks_sight(map_x, map_y):
                return False
        return True

    def cast_wall_dist(self, x0, y0, dx, dy, max_dist=64.0):
        """Distance along ray until a sight-blocking wall (for hitscan)."""
        map_x, map_y = int(x0), int(y0)
        delta_x = abs(1 / dx) if dx else 1e30
        delta_y = abs(1 / dy) if dy else 1e30
        step_x = 1 if dx > 0 else -1
        step_y = 1 if dy > 0 else -1
        side_x = ((map_x + 1 - x0) if dx > 0 else (x0 - map_x)) * delta_x
        side_y = ((map_y + 1 - y0) if dy > 0 else (y0 - map_y)) * delta_y
        while True:
            if side_x < side_y:
                d = side_x
                side_x += delta_x
                map_x += step_x
            else:
                d = side_y
                side_y += delta_y
                map_y += step_y
            if d > max_dist:
                return max_dist
            if self.blocks_sight(map_x, map_y):
                return d

    def bfs_dir(self, from_xy, to_xy):
        """First-step direction from one cell toward another (None if no path)."""
        start = (int(from_xy[0]), int(from_xy[1]))
        goal = (int(to_xy[0]), int(to_xy[1]))
        if start == goal:
            return None
        seen = {goal}
        q = deque([(goal, None)])
        while q:
            (cx, cy), first = q.popleft()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nxt = (cx + dx, cy + dy)
                if nxt in seen:
                    continue
                gx, gy = nxt
                t = self.wall_at(gx, gy)
                if t != 0 and (gx, gy) not in self.doors:
                    continue
                seen.add(nxt)
                # search runs goal->start, so the step back toward the goal
                # from `nxt` is (-dx, -dy)
                step = first if first is not None else (-dx, -dy)
                if nxt == start:
                    return step
                if len(seen) < 900:
                    q.append((nxt, step))
        return None

    def update_doors(self, dt, occupied_cells):
        for (x, y), door in self.doors.items():
            door.update(dt, blocked=(x, y) in occupied_cells)

    def press_exit(self):
        if self.exit_pos:
            x, y = self.exit_pos
            self.grid[y][x] = EXIT_PRESSED_ID
