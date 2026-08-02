"""Game entities: enemies, pickups, projectiles, barrels and visual effects."""
import math
import random

from settings import AMMO_MAX, PLAYER_MAX_HEALTH, PLAYER_MAX_ARMOR

rng = random.Random(90210)


class Entity:
    """Anything rendered as a billboard sprite in the world."""
    scale = 0.6          # height as fraction of wall height
    v_anchor = "floor"   # 'floor' or 'center'
    blocking = False
    radius = 0.3

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.alive = True

    def sprite(self, assets):
        raise NotImplementedError

    def update(self, game, dt):
        pass


# ---------------------------------------------------------------------------
# Pickups
# ---------------------------------------------------------------------------
class Pickup(Entity):
    scale = 0.28

    def __init__(self, x, y, kind):
        super().__init__(x, y)
        self.kind = kind

    SPRITES = {
        "stim": "stim", "medkit": "medkit", "clip": "clip",
        "shells": "shellbox", "cells": "cellpack", "armor": "armor",
        "key_blue": "key_blue", "key_red": "key_red",
        "shotgun": "wp_shotgun", "chaingun": "wp_chaingun",
        "plasma": "wp_plasma",
    }

    def sprite(self, assets):
        return assets[self.SPRITES[self.kind]]

    def try_collect(self, game):
        p = game.player
        k = self.kind
        msg, snd = None, "pickup"
        if k == "stim":
            if p.health >= PLAYER_MAX_HEALTH:
                return False
            p.health = min(PLAYER_MAX_HEALTH, p.health + 10)
            msg = "Picked up a stimpack."
        elif k == "medkit":
            if p.health >= PLAYER_MAX_HEALTH:
                return False
            p.health = min(PLAYER_MAX_HEALTH, p.health + 25)
            msg = "Picked up a medikit."
        elif k == "armor":
            if p.armor >= PLAYER_MAX_ARMOR:
                return False
            p.armor = PLAYER_MAX_ARMOR
            msg = "Picked up combat armor!"
        elif k == "clip":
            if not p.add_ammo("bullets", 10):
                return False
            msg = "Picked up a clip."
        elif k == "shells":
            if not p.add_ammo("shells", 8):
                return False
            msg = "Picked up a box of shells."
        elif k == "cells":
            if not p.add_ammo("cells", 40):
                return False
            msg = "Picked up an energy cell pack."
        elif k == "key_blue":
            p.keys.add("blue")
            msg, snd = "Picked up the BLUE keycard!", "key"
        elif k == "key_red":
            p.keys.add("red")
            msg, snd = "Picked up the RED keycard!", "key"
        elif k in ("shotgun", "chaingun", "plasma"):
            first = k not in p.weapons
            p.weapons.add(k)
            ammo_type, amt = {"shotgun": ("shells", 8),
                              "chaingun": ("bullets", 20),
                              "plasma": ("cells", 40)}[k]
            gained = p.add_ammo(ammo_type, amt)
            if not first and not gained:
                return False
            if first:
                p.switch_to(k)
                msg = "You got the %s!" % {"shotgun": "shotgun",
                                           "chaingun": "minigun",
                                           "plasma": "plasma rifle"}[k]
            else:
                msg = "Picked up ammo."
        if msg:
            game.message(msg)
            game.sounds.play(snd)
            game.items_got += 1
        return True


# ---------------------------------------------------------------------------
# Barrel
# ---------------------------------------------------------------------------
class Barrel(Entity):
    scale = 0.5
    blocking = True
    radius = 0.3

    def __init__(self, x, y):
        super().__init__(x, y)
        self.hp = 20

    def sprite(self, assets):
        return assets["barrel"]

    def damage(self, game, amount):
        if not self.alive:
            return
        self.hp -= amount
        if self.hp <= 0:
            self.alive = False
            game.explode(self.x, self.y, dmg=90, radius=2.1, big=True)


# ---------------------------------------------------------------------------
# Projectiles & effects
# ---------------------------------------------------------------------------
class Projectile(Entity):
    scale = 0.22
    v_anchor = "center"

    def __init__(self, x, y, dx, dy, speed, dmg, sprite_key, from_player):
        super().__init__(x, y)
        self.min_draw_dist = 0.55 if from_player else 0.0
        self.dx, self.dy = dx, dy
        self.speed = speed
        self.dmg = dmg
        self.sprite_key = sprite_key
        self.from_player = from_player
        self.anim = 0.0

    def sprite(self, assets):
        frames = assets[self.sprite_key]
        return frames[int(self.anim * 10) % len(frames)]

    def update(self, game, dt):
        self.anim += dt
        steps = max(1, int(self.speed * dt / 0.1))
        for _ in range(steps):
            self.x += self.dx * self.speed * dt / steps
            self.y += self.dy * self.speed * dt / steps
            if game.level.blocks_sight(int(self.x), int(self.y)):
                self._impact(game)
                return
            if self.from_player:
                for e in game.enemies:
                    if e.alive and (e.x - self.x) ** 2 + (e.y - self.y) ** 2 < 0.25:
                        e.damage(game, self.dmg)
                        self._impact(game)
                        return
                for b in game.barrels:
                    if b.alive and (b.x - self.x) ** 2 + (b.y - self.y) ** 2 < 0.2:
                        b.damage(game, self.dmg)
                        self._impact(game)
                        return
            else:
                p = game.player
                if (p.x - self.x) ** 2 + (p.y - self.y) ** 2 < 0.30:
                    p.hurt(game, rng.randint(*self.dmg) if isinstance(self.dmg, tuple) else self.dmg)
                    self._impact(game)
                    return

    def _impact(self, game):
        self.alive = False
        game.effects.append(Effect(self.x, self.y, "small_explosion", 0.3, scale=0.35))


class Effect(Entity):
    v_anchor = "center"

    def __init__(self, x, y, sprite_key, duration, scale=0.9):
        super().__init__(x, y)
        self.sprite_key = sprite_key
        self.duration = duration
        self.t = 0.0
        self.scale = scale

    def sprite(self, assets):
        frames = assets[self.sprite_key]
        i = min(len(frames) - 1, int(self.t / self.duration * len(frames)))
        return frames[i]

    def update(self, game, dt):
        self.t += dt
        if self.t >= self.duration:
            self.alive = False


# ---------------------------------------------------------------------------
# Enemies
# ---------------------------------------------------------------------------
ENEMY_STATS = {
    "grunt": dict(hp=30, speed=1.7, scale=0.72, radius=0.3, pain_chance=0.65,
                  attack_range=11.0, attack_time=0.55, attack_hit=0.30,
                  cooldown=(0.8, 1.8), melee=False, projectile=None,
                  dmg=(3, 12), sight_sound="growl_hi"),
    "fiend": dict(hp=60, speed=2.1, scale=0.78, radius=0.32, pain_chance=0.5,
                  attack_range=10.0, attack_time=0.6, attack_hit=0.35,
                  cooldown=(1.0, 2.0), melee=False,
                  projectile=dict(speed=6.5, dmg=(8, 18), sprite="fireball"),
                  dmg=None, sight_sound="growl"),
    "ravager": dict(hp=160, speed=2.5, scale=0.86, radius=0.4, pain_chance=0.3,
                    attack_range=1.25, attack_time=0.5, attack_hit=0.28,
                    cooldown=(0.5, 1.0), melee=True, projectile=None,
                    dmg=(12, 28), sight_sound="growl"),
}


class Enemy(Entity):
    blocking = True

    def __init__(self, x, y, kind):
        super().__init__(x, y)
        self.kind = kind
        st = ENEMY_STATS[kind]
        self.hp = st["hp"]
        self.speed = st["speed"]
        self.scale = st["scale"]
        self.radius = st["radius"]
        self.state = "idle"          # idle/chase/attack/pain/dying/dead
        self.state_t = 0.0
        self.anim_t = rng.random()
        self.cool = 0.0
        self.attack_done = False
        self.repath = 0.0
        self.move_dir = None

    # -- combat -----------------------------------------------------------

    def damage(self, game, amount):
        if not self.alive or self.state == "dying":
            return
        self.hp -= amount
        self.alert()
        st = ENEMY_STATS[self.kind]
        if self.hp <= 0:
            self.state = "dying"
            self.state_t = 0.0
            game.sounds.play("death")
            game.kills += 1
        elif rng.random() < st["pain_chance"]:
            self.state = "pain"
            self.state_t = 0.0
            game.sounds.play("pain")

    def alert(self):
        if self.state == "idle":
            self.state = "chase"

    # -- brain ------------------------------------------------------------

    def update(self, game, dt):
        if self.state == "dead":
            return
        self.state_t += dt
        self.anim_t += dt
        self.cool = max(0.0, self.cool - dt)
        st = ENEMY_STATS[self.kind]
        p = game.player
        dist = math.hypot(p.x - self.x, p.y - self.y)

        if self.state == "dying":
            if self.state_t > 0.6:
                self.state = "dead"
                self.blocking = False
            return

        if self.state == "pain":
            if self.state_t > 0.35:
                self.state = "chase"
            return

        los = game.level.line_of_sight(self.x, self.y, p.x, p.y)

        if self.state == "idle":
            if los and dist < 14:
                self.state = "chase"
                game.sounds.play(st["sight_sound"])
            return

        if self.state == "attack":
            if not self.attack_done and self.state_t >= st["attack_time"] * 0.6:
                self.attack_done = True
                self._deal_attack(game, dist, los)
            if self.state_t >= st["attack_time"]:
                self.state = "chase"
            return

        # chase
        want_attack = (self.cool <= 0.0 and
                       ((st["melee"] and dist < st["attack_range"]) or
                        (not st["melee"] and los and dist < st["attack_range"]
                         and rng.random() < 0.03 + dt * 2)))
        if want_attack:
            self.state = "attack"
            self.state_t = 0.0
            self.attack_done = False
            self.cool = rng.uniform(*st["cooldown"])
            return

        # movement
        if dist > (st["attack_range"] * 0.5 if not st["melee"] else 0.8):
            if los:
                mdx, mdy = (p.x - self.x) / dist, (p.y - self.y) / dist
            else:
                self.repath -= dt
                if self.repath <= 0 or self.move_dir is None:
                    self.repath = 0.4 + rng.random() * 0.3
                    self.move_dir = game.level.bfs_dir((self.x, self.y), (p.x, p.y))
                if self.move_dir is None:
                    return
                tx = int(self.x) + self.move_dir[0] + 0.5
                ty = int(self.y) + self.move_dir[1] + 0.5
                d = math.hypot(tx - self.x, ty - self.y) or 1.0
                mdx, mdy = (tx - self.x) / d, (ty - self.y) / d
            blockers = [(e.x, e.y, e.radius) for e in game.enemies
                        if e is not self and e.alive and e.blocking]
            blockers += [(b.x, b.y, b.radius) for b in game.barrels if b.alive]
            self.x, self.y = game.level.move_circle(
                self.x, self.y, mdx * self.speed * dt, mdy * self.speed * dt,
                self.radius, blockers)

    def _deal_attack(self, game, dist, los):
        st = ENEMY_STATS[self.kind]
        p = game.player
        if st["melee"]:
            if dist < st["attack_range"] * 1.3:
                p.hurt(game, rng.randint(*st["dmg"]))
        elif st["projectile"]:
            pr = st["projectile"]
            d = math.hypot(p.x - self.x, p.y - self.y) or 1.0
            game.projectiles.append(Projectile(
                self.x, self.y, (p.x - self.x) / d, (p.y - self.y) / d,
                pr["speed"], pr["dmg"], pr["sprite"], from_player=False))
            game.sounds.play("fireball")
        else:
            game.sounds.play("pistol")
            if los:
                chance = max(0.12, 0.75 - dist * 0.05)
                if rng.random() < chance:
                    p.hurt(game, rng.randint(*st["dmg"]))

    # -- drawing ----------------------------------------------------------

    def sprite(self, assets):
        frames = assets[self.kind]
        if self.state == "dead":
            return frames["corpse"]
        if self.state == "dying":
            death = frames["death"]
            i = min(len(death) - 1, int(self.state_t / 0.6 * len(death)))
            return death[i]
        if self.state == "pain":
            return frames["pain"][0]
        if self.state == "attack":
            return frames["attack"][0]
        if self.state == "chase":
            return frames["walk"][int(self.anim_t * 4) % 2]
        return frames["walk"][0]


SPAWN_TABLE = {
    "g": ("enemy", "grunt"), "i": ("enemy", "fiend"), "v": ("enemy", "ravager"),
    "h": ("pickup", "stim"), "H": ("pickup", "medkit"),
    "a": ("pickup", "clip"), "s": ("pickup", "shells"), "c": ("pickup", "cells"),
    "A": ("pickup", "armor"), "u": ("pickup", "key_blue"), "r": ("pickup", "key_red"),
    "2": ("pickup", "shotgun"), "3": ("pickup", "chaingun"), "5": ("pickup", "plasma"),
    "o": ("barrel", None),
}
