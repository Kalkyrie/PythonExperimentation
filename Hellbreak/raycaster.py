"""The renderer: numpy floor/ceiling casting, DDA wall raycasting with
sliding-door support, and z-buffered billboard sprites."""
import math

import numpy as np
import pygame

from settings import (RENDER_W, RENDER_H, HALF_H, TEX_SIZE, PLANE_LEN,
                      FOG_STRENGTH, MIN_BRIGHT, SIDE_SHADE, MAX_SPRITE_H)

W, H = RENDER_W, RENDER_H
T = TEX_SIZE


def brightness(dist):
    return max(MIN_BRIGHT, min(1.0, 1.15 / (1.0 + dist * dist * FOG_STRENGTH)))


class Raycaster:
    def __init__(self, walls, wall_cols, floors, ceils):
        self.walls = walls
        self.wall_cols = wall_cols
        self.floor_arrays = floors
        self.ceil_arrays = ceils
        self.surface = pygame.Surface((W, H))
        self.zbuffer = [1e30] * W
        # precomputed values for the floor caster
        ys = np.arange(HALF_H + 1, H)
        self.row_dist = (0.5 * H) / (ys - HALF_H)          # (rows,)
        shade = 1.15 / (1.0 + self.row_dist ** 2 * FOG_STRENGTH)
        self.row_shade = np.clip(shade, MIN_BRIGHT, 1.0)[:, None]  # (rows,1)
        self.col_frac = np.arange(W) / W                    # (W,)
        self.fc_buffer = np.zeros((H, W, 3), dtype=np.uint8)

    # ------------------------------------------------------------------
    def render(self, level, px, py, angle):
        dir_x, dir_y = math.cos(angle), math.sin(angle)
        plane_x, plane_y = -dir_y * PLANE_LEN, dir_x * PLANE_LEN

        self._floor_ceiling(level, px, py, dir_x, dir_y, plane_x, plane_y)
        self._walls(level, px, py, dir_x, dir_y, plane_x, plane_y)
        return self.surface

    # ------------------------------------------------------------------
    def _floor_ceiling(self, level, px, py, dir_x, dir_y, plane_x, plane_y):
        ray0x, ray0y = dir_x - plane_x, dir_y - plane_y
        ray1x, ray1y = dir_x + plane_x, dir_y + plane_y
        rd = self.row_dist[:, None]                        # (rows,1)
        fx = px + rd * (ray0x + (ray1x - ray0x) * self.col_frac)  # (rows,W)
        fy = py + rd * (ray0y + (ray1y - ray0y) * self.col_frac)
        tx = (np.abs(fx * T).astype(np.int32)) & (T - 1)
        ty = (np.abs(fy * T).astype(np.int32)) & (T - 1)
        cell_parity = (fx.astype(np.int32) + fy.astype(np.int32)) & 1

        f1 = self.floor_arrays[level.floor_tex]
        f2 = self.floor_arrays[level.floor2_tex]
        floor_px = np.where(cell_parity[..., None] == 0,
                            f1[ty, tx], f2[ty, tx]).astype(np.float32)
        floor_px *= self.row_shade[..., None]

        c = self.ceil_arrays[level.ceil_tex]
        ceil_px = c[ty, tx].astype(np.float32) * self.row_shade[..., None]

        buf = self.fc_buffer
        buf[HALF_H + 1:H] = floor_px.astype(np.uint8)
        buf[0:H - HALF_H - 1] = ceil_px[::-1].astype(np.uint8)
        buf[HALF_H - 1:HALF_H + 1] = 12
        pygame.surfarray.blit_array(self.surface, np.transpose(buf, (1, 0, 2)))

    # ------------------------------------------------------------------
    def _walls(self, level, px, py, dir_x, dir_y, plane_x, plane_y):
        surf = self.surface
        zbuf = self.zbuffer
        grid = level.grid
        doors = level.doors
        gw, gh = level.w, level.h

        for x in range(W):
            camera_x = 2.0 * x / W - 1.0
            rdx = dir_x + plane_x * camera_x
            rdy = dir_y + plane_y * camera_x
            map_x, map_y = int(px), int(py)
            delta_x = abs(1.0 / rdx) if rdx else 1e30
            delta_y = abs(1.0 / rdy) if rdy else 1e30
            if rdx < 0:
                step_x, side_x = -1, (px - map_x) * delta_x
            else:
                step_x, side_x = 1, (map_x + 1.0 - px) * delta_x
            if rdy < 0:
                step_y, side_y = -1, (py - map_y) * delta_y
            else:
                step_y, side_y = 1, (map_y + 1.0 - py) * delta_y

            tex_id, side, perp = 0, 0, 1e30
            is_door = False
            for _ in range(96):
                if side_x < side_y:
                    side_x += delta_x
                    map_x += step_x
                    side = 0
                else:
                    side_y += delta_y
                    map_y += step_y
                    side = 1
                if not (0 <= map_x < gw and 0 <= map_y < gh):
                    tex_id = 1
                    perp = (side_x - delta_x) if side == 0 else (side_y - delta_y)
                    break
                tid = grid[map_y][map_x]
                if tid == 0:
                    continue
                door = doors.get((map_x, map_y))
                if door is not None:
                    # door plane sits at the middle of the cell
                    if side == 0:
                        d = side_x - delta_x * 0.5
                    else:
                        d = side_y - delta_y * 0.5
                    hx = px + rdx * d
                    hy = py + rdy * d
                    if side == 0:
                        wall_u = hy - map_y
                        in_cell = 0.0 <= wall_u < 1.0
                    else:
                        wall_u = hx - map_x
                        in_cell = 0.0 <= wall_u < 1.0
                    if not in_cell or wall_u < door.open:
                        continue  # ray slips through the opening / past jamb
                    tex_id, perp, is_door = tid, d, True
                    door_u = wall_u - door.open
                    break
                tex_id = tid
                perp = (side_x - delta_x) if side == 0 else (side_y - delta_y)
                break

            if tex_id == 0 or perp <= 1e-6:
                zbuf[x] = 1e30
                continue
            zbuf[x] = perp

            if is_door:
                wall_u = door_u
            else:
                if side == 0:
                    wall_u = py + perp * rdy
                else:
                    wall_u = px + perp * rdx
                wall_u -= math.floor(wall_u)
            tex_x = int(wall_u * T)
            if (side == 0 and rdx > 0) or (side == 1 and rdy < 0):
                tex_x = T - tex_x - 1
            tex_x = max(0, min(T - 1, tex_x))

            line_h = int(H / perp)
            cols = self.wall_cols[tex_id][1 if side == 1 else 0]
            col_surf = cols[tex_x]
            if line_h <= H:
                draw_y = HALF_H - line_h // 2
                scaled = pygame.transform.scale(col_surf, (1, line_h))
                surf.blit(scaled, (x, draw_y))
                shade_h = line_h
                shade_y = draw_y
            else:
                # only scale the visible slice of the texture column
                line_h = min(line_h, MAX_SPRITE_H)
                top = (line_h - H) / 2
                ty0 = int(top * T / line_h)
                ty1 = T - ty0
                if ty1 <= ty0:
                    ty1 = ty0 + 1
                sub = col_surf.subsurface(0, ty0, 1, ty1 - ty0)
                scaled = pygame.transform.scale(sub, (1, H))
                surf.blit(scaled, (x, 0))
                shade_h = H
                shade_y = 0

            b = brightness(perp)
            if b < 0.99:
                v = int(b * 255)
                surf.fill((v, v, v), rect=(x, shade_y, 1, shade_h),
                          special_flags=pygame.BLEND_RGB_MULT)

    # ------------------------------------------------------------------
    def draw_sprites(self, entities, assets, px, py, angle):
        """Render entity billboards back-to-front with per-column z-test."""
        dir_x, dir_y = math.cos(angle), math.sin(angle)
        plane_x, plane_y = -dir_y * PLANE_LEN, dir_x * PLANE_LEN
        inv_det = 1.0 / (plane_x * dir_y - dir_x * plane_y)
        surf = self.surface
        zbuf = self.zbuffer

        order = []
        for e in entities:
            dist2 = (e.x - px) ** 2 + (e.y - py) ** 2
            order.append((dist2, e))
        order.sort(key=lambda t: -t[0])

        for _, e in order:
            sx = e.x - px
            sy = e.y - py
            trans_x = inv_det * (dir_y * sx - dir_x * sy)
            trans_y = inv_det * (-plane_y * sx + plane_x * sy)
            if trans_y <= max(0.08, getattr(e, "min_draw_dist", 0.0)):
                continue
            screen_x = int((W / 2) * (1 + trans_x / trans_y))

            img = e.sprite(assets)
            iw, ih = img.get_size()
            wall_h = H / trans_y
            sh = int(wall_h * e.scale)
            if sh < 2:
                continue
            sh = min(sh, MAX_SPRITE_H)
            sw = max(2, int(sh * iw / ih))
            if e.v_anchor == "floor":
                v_center = HALF_H + wall_h / 2 - sh / 2
            else:
                v_center = HALF_H + wall_h * 0.08
            top = int(v_center - sh / 2)
            left = screen_x - sw // 2
            if left + sw < 0 or left >= W:
                continue

            scaled = pygame.transform.scale(img, (sw, sh))
            b = brightness(trans_y)
            if b < 0.99:
                v = int(b * 255)
                scaled = scaled.copy()
                scaled.fill((v, v, v), special_flags=pygame.BLEND_RGB_MULT)

            x0 = max(0, left)
            x1 = min(W, left + sw)
            col = x0
            while col < x1:
                # batch adjacent visible columns into one blit
                if zbuf[col] > trans_y:
                    run = col
                    while run < x1 and zbuf[run] > trans_y:
                        run += 1
                    area = pygame.Rect(col - left, 0, run - col, sh)
                    surf.blit(scaled, (col, top), area)
                    col = run
                else:
                    col += 1
