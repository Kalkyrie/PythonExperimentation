"""HUD: status bar, first-person weapon, crosshair, minimap, messages."""
import math

import pygame

from settings import (WIN_W, WIN_H, SCALE, COL_HUD_BG, COL_HUD_EDGE,
                      COL_AMBER, COL_RED, COL_GREEN, COL_BLUE, COL_TEXT)
from player import WEAPONS

BAR_H = 36 * SCALE // 3 * 3          # status bar height in window pixels


class Hud:
    def __init__(self, assets):
        self.assets = assets
        pygame.font.init()
        self.font_big = pygame.font.Font(None, 22)
        self.font_small = pygame.font.Font(None, 16)
        self.msg = ""
        self.msg_t = 0.0

    def message(self, text):
        self.msg = text
        self.msg_t = 3.0

    def update(self, dt):
        self.msg_t = max(0.0, self.msg_t - dt)

    # ------------------------------------------------------------------
    def _text(self, surf, text, x, y, color=COL_TEXT, big=True, center=False):
        font = self.font_big if big else self.font_small
        img = font.render(text, False, color)
        img = pygame.transform.scale(img, (img.get_width() * 2, img.get_height() * 2))
        r = img.get_rect()
        if center:
            r.center = (x, y)
        else:
            r.topleft = (x, y)
        surf.blit(img, r)

    # ------------------------------------------------------------------
    def draw_weapon(self, win, player, bob_phase):
        w = WEAPONS[player.current]
        img = self.assets[w["view"]]
        bob_x = math.sin(bob_phase) * 14 if player.moving else 0
        bob_y = abs(math.cos(bob_phase)) * 10 if player.moving else 0
        kick = 0
        if player.fire_anim > 0:
            kick = int(w["kick"] * (player.fire_anim / 0.25))
        if player.pending:
            kick += int(120 * (0.25 - player.switch_t) / 0.25) if player.switch_t > 0.125 \
                else int(120 * player.switch_t / 0.125)
        x = WIN_W // 2 - img.get_width() // 2 + int(bob_x)
        y = WIN_H - BAR_H - img.get_height() + 26 + int(bob_y) + kick
        if player.fire_anim > 0.1:
            mz = self.assets["muzzle"]
            mz = pygame.transform.scale(mz, (140, 140))
            win.blit(mz, (WIN_W // 2 - 70 + int(bob_x), y - 90))
        win.blit(img, (x, y))

    # ------------------------------------------------------------------
    def draw_bar(self, win, player, game):
        y0 = WIN_H - BAR_H
        pygame.draw.rect(win, COL_HUD_BG, (0, y0, WIN_W, BAR_H))
        pygame.draw.rect(win, COL_HUD_EDGE, (0, y0, WIN_W, 3))

        w = WEAPONS[player.current]
        ammo = player.ammo[w["ammo"]]
        self._text(win, "AMMO", 30, y0 + 8, COL_TEXT, big=False)
        self._text(win, f"{ammo}", 30, y0 + 34, COL_AMBER)
        self._text(win, "HEALTH", 170, y0 + 8, COL_TEXT, big=False)
        hp_col = COL_GREEN if player.health > 50 else \
            (COL_AMBER if player.health > 25 else COL_RED)
        self._text(win, f"{player.health}%", 170, y0 + 34, hp_col)
        self._text(win, "ARMOR", 330, y0 + 8, COL_TEXT, big=False)
        self._text(win, f"{player.armor}%", 330, y0 + 34, COL_BLUE)

        # face
        if player.health > 60:
            face = self.assets["face_ok"]
        elif player.health > 25:
            face = self.assets["face_hurt"]
        else:
            face = self.assets["face_bad"]
        win.blit(face, (WIN_W // 2 - face.get_width() // 2 + 40, y0 + 12))

        # keys
        kx = WIN_W - 320
        self._text(win, "KEYS", kx, y0 + 8, COL_TEXT, big=False)
        for i, key in enumerate(("blue", "red")):
            if key in player.keys:
                img = self.assets["key_%s" % key]
                win.blit(img, (kx + i * 30, y0 + 34))

        # ammo stores
        sx = WIN_W - 210
        for i, (label, kind) in enumerate((("BULL", "bullets"),
                                           ("SHEL", "shells"),
                                           ("CELL", "cells"))):
            self._text(win, f"{label} {player.ammo[kind]}", sx, y0 + 8 + i * 26,
                       COL_TEXT, big=False)

        # weapon slots owned
        slots = "".join(str(c["slot"]) if n in player.weapons else "-"
                        for n, c in sorted(WEAPONS.items(), key=lambda kv: kv[1]["slot"]))
        self._text(win, slots, 30, y0 + 76, (140, 135, 125), big=False)

    # ------------------------------------------------------------------
    def draw_crosshair(self, win):
        cx, cy = WIN_W // 2, (WIN_H - BAR_H) // 2
        pygame.draw.line(win, (210, 210, 200), (cx - 7, cy), (cx - 3, cy), 2)
        pygame.draw.line(win, (210, 210, 200), (cx + 3, cy), (cx + 7, cy), 2)
        pygame.draw.line(win, (210, 210, 200), (cx, cy - 7), (cx, cy - 3), 2)
        pygame.draw.line(win, (210, 210, 200), (cx, cy + 3), (cx, cy + 7), 2)

    # ------------------------------------------------------------------
    def draw_messages(self, win, game):
        if self.msg_t > 0:
            self._text(win, self.msg, 14, 12, COL_AMBER, big=False)

    # ------------------------------------------------------------------
    def draw_minimap(self, win, game):
        level, p = game.level, game.player
        cell = 6
        pad = 10
        mw, mh = level.w * cell, level.h * cell
        surf = pygame.Surface((mw, mh), pygame.SRCALPHA)
        surf.fill((10, 10, 12, 170))
        for y in range(level.h):
            for x in range(level.w):
                t = level.grid[y][x]
                if t == 0:
                    continue
                if (x, y) in level.doors:
                    col = (200, 170, 60)
                elif t == 10:
                    col = (220, 60, 50)
                else:
                    col = (120, 115, 110)
                surf.fill(col, (x * cell, y * cell, cell - 1, cell - 1))
        for e in game.enemies:
            if e.alive and e.state != "dead":
                surf.fill((200, 40, 30), (int(e.x * cell) - 2, int(e.y * cell) - 2, 4, 4))
        px, py = int(p.x * cell), int(p.y * cell)
        pygame.draw.circle(surf, (80, 220, 80), (px, py), 3)
        dx, dy = p.dir
        pygame.draw.line(surf, (80, 220, 80), (px, py),
                         (px + int(dx * 10), py + int(dy * 10)), 2)
        win.blit(surf, (WIN_W - mw - pad, pad))
