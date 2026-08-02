"""Player state, movement, and weapon handling."""
import math
import random

from settings import (PLAYER_RADIUS, MOVE_SPEED, RUN_MULT, TURN_SPEED,
                      PLAYER_MAX_HEALTH, AMMO_MAX, USE_RANGE)
from entities import Projectile, Effect

rng = random.Random(31337)

WEAPONS = {
    "pistol":   dict(slot=1, ammo="bullets", cost=1, pellets=1, dmg=(5, 14),
                     rate=0.42, spread=0.025, auto=False, sound="pistol",
                     view="view_pistol", kick=6),
    "shotgun":  dict(slot=2, ammo="shells", cost=1, pellets=7, dmg=(4, 9),
                     rate=0.95, spread=0.10, auto=False, sound="shotgun",
                     view="view_shotgun", kick=14),
    "chaingun": dict(slot=3, ammo="bullets", cost=1, pellets=1, dmg=(5, 12),
                     rate=0.11, spread=0.055, auto=True, sound="chaingun",
                     view="view_chaingun", kick=4),
    "plasma":   dict(slot=4, ammo="cells", cost=1, pellets=0, dmg=22,
                     rate=0.17, spread=0.0, auto=True, sound="plasma",
                     view="view_plasma", kick=5,
                     projectile=dict(speed=11.0, sprite="plasmaball")),
}
SLOT_TO_WEAPON = {w["slot"]: name for name, w in WEAPONS.items()}


class Player:
    def __init__(self, x, y, angle):
        self.x, self.y = x, y
        self.angle = angle
        self.health = PLAYER_MAX_HEALTH
        self.armor = 0
        self.ammo = {"bullets": 50, "shells": 0, "cells": 0}
        self.weapons = {"pistol"}
        self.keys = set()
        self.current = "pistol"
        self.pending = None
        self.fire_timer = 0.0
        self.fire_anim = 0.0
        self.switch_t = 0.0
        self.bob = 0.0
        self.moving = False
        self.dead = False

    # -- helpers ----------------------------------------------------------

    @property
    def dir(self):
        return math.cos(self.angle), math.sin(self.angle)

    def add_ammo(self, kind, amount):
        if self.ammo[kind] >= AMMO_MAX[kind]:
            return False
        self.ammo[kind] = min(AMMO_MAX[kind], self.ammo[kind] + amount)
        return True

    def switch_to(self, name):
        if name in self.weapons and name != self.current:
            self.pending = name
            self.switch_t = 0.25

    # -- per-frame --------------------------------------------------------

    def update(self, game, dt, keys_held, turn_input, mouse_dx, fire_held):
        if self.dead:
            return
        self.fire_timer = max(0.0, self.fire_timer - dt)
        self.fire_anim = max(0.0, self.fire_anim - dt)
        if self.pending:
            self.switch_t -= dt
            if self.switch_t <= 0:
                self.current = self.pending
                self.pending = None

        self.angle = (self.angle + turn_input * TURN_SPEED * dt +
                      mouse_dx) % (2 * math.pi)

        dx = dy = 0.0
        cos_a, sin_a = math.cos(self.angle), math.sin(self.angle)
        if keys_held["fwd"]:
            dx += cos_a
            dy += sin_a
        if keys_held["back"]:
            dx -= cos_a
            dy -= sin_a
        if keys_held["left"]:
            dx += sin_a
            dy -= cos_a
        if keys_held["right"]:
            dx -= sin_a
            dy += cos_a
        mag = math.hypot(dx, dy)
        self.moving = mag > 1e-6
        if self.moving:
            speed = MOVE_SPEED * (RUN_MULT if keys_held["run"] else 1.0)
            dx, dy = dx / mag * speed * dt, dy / mag * speed * dt
            blockers = [(e.x, e.y, e.radius) for e in game.enemies
                        if e.alive and e.blocking]
            blockers += [(b.x, b.y, b.radius) for b in game.barrels if b.alive]
            self.x, self.y = game.level.move_circle(
                self.x, self.y, dx, dy, PLAYER_RADIUS, blockers)
            self.bob += dt * (10 if keys_held["run"] else 7)

        if fire_held and self.fire_timer <= 0 and not self.pending:
            self.fire(game)

    # -- combat -----------------------------------------------------------

    def fire(self, game):
        w = WEAPONS[self.current]
        if self.ammo[w["ammo"]] < w["cost"]:
            game.sounds.play("click")
            self.fire_timer = 0.3
            # auto-fallback to pistol
            if self.current != "pistol":
                self.switch_to("pistol")
            return
        self.ammo[w["ammo"]] -= w["cost"]
        self.fire_timer = w["rate"]
        self.fire_anim = min(0.25, w["rate"])
        game.sounds.play(w["sound"])
        game.alert_enemies(self.x, self.y, radius=13)

        if "projectile" in w:
            pr = w["projectile"]
            cos_a, sin_a = math.cos(self.angle), math.sin(self.angle)
            game.projectiles.append(Projectile(
                self.x + cos_a * 0.6, self.y + sin_a * 0.6, cos_a, sin_a,
                pr["speed"], w["dmg"], pr["sprite"], from_player=True))
            return

        for _ in range(w["pellets"]):
            ang = self.angle + rng.uniform(-w["spread"], w["spread"])
            self._hitscan(game, ang, rng.randint(*w["dmg"]))

    def _hitscan(self, game, angle, dmg):
        dx, dy = math.cos(angle), math.sin(angle)
        wall_d = game.level.cast_wall_dist(self.x, self.y, dx, dy)
        best, best_t = None, wall_d
        for group in (game.enemies, game.barrels):
            for e in group:
                if not e.alive or getattr(e, "state", "") in ("dying", "dead"):
                    continue
                ex, ey = e.x - self.x, e.y - self.y
                t = ex * dx + ey * dy          # distance along the ray
                if t < 0.2 or t > best_t:
                    continue
                perp = abs(ex * dy - ey * dx)  # distance from ray to centre
                if perp < e.radius + 0.12:
                    best, best_t = e, t
        if best is not None:
            best.damage(game, dmg)
            game.effects.append(Effect(best.x, best.y, "small_explosion",
                                       0.15, scale=0.18))
        else:
            hx, hy = self.x + dx * (wall_d - 0.05), self.y + dy * (wall_d - 0.05)
            game.effects.append(Effect(hx, hy, "small_explosion", 0.12, scale=0.10))

    def hurt(self, game, dmg):
        if self.dead:
            return
        absorbed = min(self.armor, dmg // 3)
        self.armor -= absorbed
        self.health -= (dmg - absorbed)
        game.damage_flash = min(1.0, game.damage_flash + dmg / 40)
        game.sounds.play("pain")
        if self.health <= 0:
            self.health = 0
            self.dead = True

    # -- interaction ------------------------------------------------------

    def use(self, game):
        cos_a, sin_a = math.cos(self.angle), math.sin(self.angle)
        for dist in (0.6, 1.0, USE_RANGE + 0.3):
            tx, ty = int(self.x + cos_a * dist), int(self.y + sin_a * dist)
            if (tx, ty) == (int(self.x), int(self.y)):
                continue
            tile = game.level.wall_at(tx, ty)
            if tile == 0:
                continue
            door = game.level.doors.get((tx, ty))
            if door is not None:
                if door.key and door.key not in self.keys:
                    game.message("You need the %s keycard." % door.key.upper())
                    game.sounds.play("click")
                elif door.state in ("closed", "closing"):
                    door.state = "opening"
                    game.sounds.play("door")
                return
            if tile == 10:  # exit switch
                game.sounds.play("switch")
                game.level.press_exit()
                game.finish_level()
                return
            return  # plain wall: nothing to do
