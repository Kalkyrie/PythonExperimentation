"""HELLBREAK — a retro raycasting FPS in the spirit of the 1994 classics.

Run:  python main.py

Controls
--------
  W/A/S/D ......... move / strafe      Mouse ....... turn (click to grab)
  Left/Right ...... turn (keyboard)    LMB / Ctrl .. fire
  Shift ........... run                E / Space ... open doors, use switches
  1-4 ............. select weapon      Tab ......... toggle minimap
  Esc ............. pause / release mouse
"""
import math
import os
import sys

import pygame

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from settings import (WIN_W, WIN_H, FPS_CAP, MOUSE_SENS,
                      COL_AMBER, COL_RED, COL_TEXT)
import texgen
import pixelart
from sounds import SoundBank
from world import Level
from mapdata import LEVELS
from entities import Enemy, Pickup, Barrel, Effect, SPAWN_TABLE
from player import Player, SLOT_TO_WEAPON
from raycaster import Raycaster
from hud import Hud


class Game:
    """One level in progress: world state + actors."""

    def __init__(self, app, level_index):
        self.app = app
        self.sounds = app.sounds
        self.level_index = level_index
        self.level = Level(LEVELS[level_index])
        px, py = self.level.player_start
        self.player = Player(px, py, self.level.player_angle)
        self.enemies = []
        self.pickups = []
        self.barrels = []
        self.projectiles = []
        self.effects = []
        self.kills = 0
        self.items_got = 0
        self.total_items = 0
        self.damage_flash = 0.0
        self.pickup_flash = 0.0
        self.finished = False
        self.time = 0.0

        for ch, x, y in self.level.spawns:
            kind, arg = SPAWN_TABLE.get(ch, (None, None))
            if kind == "enemy":
                self.enemies.append(Enemy(x, y, arg))
            elif kind == "pickup":
                self.pickups.append(Pickup(x, y, arg))
                self.total_items += 1
            elif kind == "barrel":
                self.barrels.append(Barrel(x, y))
        self.total_kills = len(self.enemies)

    # -- services used by entities ----------------------------------------

    def message(self, text):
        self.app.hud.message(text)
        self.pickup_flash = 0.25

    def alert_enemies(self, x, y, radius):
        for e in self.enemies:
            if e.alive and e.state == "idle":
                d = math.hypot(e.x - x, e.y - y)
                if d < radius and (d < 5 or
                                   self.level.line_of_sight(e.x, e.y, x, y)):
                    e.alert()

    def explode(self, x, y, dmg, radius, big=False):
        self.sounds.play("explosion")
        self.effects.append(Effect(x, y, "explosion" if big else "small_explosion",
                                   0.5, scale=1.0 if big else 0.5))
        for e in self.enemies:
            if e.alive and e.state not in ("dying", "dead"):
                d = math.hypot(e.x - x, e.y - y)
                if d < radius:
                    e.damage(self, int(dmg * (1 - d / radius)))
        for b in self.barrels:
            if b.alive:
                d = math.hypot(b.x - x, b.y - y)
                if 0.01 < d < radius:
                    b.damage(self, 40)
        d = math.hypot(self.player.x - x, self.player.y - y)
        if d < radius:
            self.player.hurt(self, int(dmg * (1 - d / radius)))

    def finish_level(self):
        self.finished = True

    # -- frame ------------------------------------------------------------

    def update(self, dt, keys_held, turn, mouse_dx, fire_held):
        self.time += dt
        self.damage_flash = max(0.0, self.damage_flash - dt * 2.2)
        self.pickup_flash = max(0.0, self.pickup_flash - dt * 2.0)
        p = self.player
        p.update(self, dt, keys_held, turn, mouse_dx, fire_held)

        occupied = {(int(p.x), int(p.y))}
        for e in self.enemies:
            if e.alive:
                occupied.add((int(e.x), int(e.y)))
        self.level.update_doors(dt, occupied)

        for e in self.enemies:
            if e.alive:
                e.update(self, dt)
        for pr in self.projectiles:
            pr.update(self, dt)
        for fx in self.effects:
            fx.update(self, dt)
        self.projectiles = [x for x in self.projectiles if x.alive]
        self.effects = [x for x in self.effects if x.alive]

        for item in self.pickups:
            if item.alive and (item.x - p.x) ** 2 + (item.y - p.y) ** 2 < 0.45:
                if item.try_collect(self):
                    item.alive = False
        self.pickups = [i for i in self.pickups if i.alive]

    def visible_entities(self):
        ents = []
        ents.extend(e for e in self.enemies if e.alive or e.state == "dead")
        ents.extend(self.pickups)
        ents.extend(b for b in self.barrels if b.alive)
        ents.extend(self.projectiles)
        ents.extend(self.effects)
        return ents


class App:
    STATE_TITLE, STATE_PLAY, STATE_PAUSE, STATE_DEAD, STATE_INTERMISSION, \
        STATE_VICTORY = range(6)

    def __init__(self, headless=False):
        os.environ.setdefault("SDL_VIDEO_CENTERED", "1")
        pygame.init()
        try:
            pygame.mixer.init(22050, -16, 1, 512)
        except pygame.error:
            pass
        pygame.display.set_caption("HELLBREAK")
        self.win = pygame.display.set_mode((WIN_W, WIN_H))
        self.clock = pygame.time.Clock()
        self.headless = headless

        walls, wall_cols, floors, ceils = texgen.build_all()
        self.assets = pixelart.build_all()
        self.caster = Raycaster(walls, wall_cols, floors, ceils)
        self.sounds = SoundBank()
        self.hud = Hud(self.assets)
        self.font_huge = pygame.font.Font(None, 40)

        self.state = self.STATE_TITLE
        self.game = None
        self.mouse_grabbed = False
        self.show_map = False
        self.running = True

    # ------------------------------------------------------------------
    def big_text(self, text, y, color=COL_TEXT, size=3):
        img = self.font_huge.render(text, False, color)
        img = pygame.transform.scale(
            img, (img.get_width() * size, img.get_height() * size))
        r = img.get_rect(center=(WIN_W // 2, y))
        self.win.blit(img, r)

    def med_text(self, text, y, color=COL_TEXT):
        self.big_text(text, y, color, size=1)

    def grab_mouse(self, grab):
        if self.headless:
            return
        self.mouse_grabbed = grab
        pygame.event.set_grab(grab)
        pygame.mouse.set_visible(not grab)

    def start_game(self, level_index=0, carry=None):
        self.game = Game(self, level_index)
        if carry:
            p = self.game.player
            p.health, p.armor = carry["health"], carry["armor"]
            p.ammo = carry["ammo"]
            p.weapons = carry["weapons"]
            p.current = carry["current"]
        self.state = self.STATE_PLAY
        self.grab_mouse(True)
        self.sounds.start_music()

    # ------------------------------------------------------------------
    def run(self):
        while self.running:
            dt = min(0.05, self.clock.tick(FPS_CAP) / 1000.0)
            events = pygame.event.get()
            for ev in events:
                if ev.type == pygame.QUIT:
                    self.running = False
            if self.state == self.STATE_PLAY:
                self.tick_play(dt, events)
            else:
                self.tick_menu(events)
            pygame.display.flip()
        pygame.quit()

    # ------------------------------------------------------------------
    def tick_menu(self, events):
        for ev in events:
            if ev.type == pygame.KEYDOWN:
                if self.state == self.STATE_TITLE:
                    if ev.key in (pygame.K_RETURN, pygame.K_SPACE):
                        self.start_game(0)
                    elif ev.key == pygame.K_ESCAPE:
                        self.running = False
                elif self.state == self.STATE_PAUSE:
                    if ev.key == pygame.K_ESCAPE:
                        self.state = self.STATE_PLAY
                        self.grab_mouse(True)
                    elif ev.key == pygame.K_q:
                        self.sounds.stop_music()
                        self.state = self.STATE_TITLE
                elif self.state == self.STATE_DEAD:
                    if ev.key in (pygame.K_RETURN, pygame.K_SPACE):
                        self.start_game(self.game.level_index)
                elif self.state == self.STATE_INTERMISSION:
                    if ev.key in (pygame.K_RETURN, pygame.K_SPACE):
                        p = self.game.player
                        carry = dict(health=p.health, armor=p.armor,
                                     ammo=p.ammo, weapons=p.weapons,
                                     current=p.current)
                        self.start_game(self.game.level_index + 1, carry)
                elif self.state == self.STATE_VICTORY:
                    if ev.key in (pygame.K_RETURN, pygame.K_SPACE):
                        self.sounds.stop_music()
                        self.state = self.STATE_TITLE

        self.draw_menu()

    def draw_menu(self):
        self.win.fill((16, 10, 10))
        if self.state == self.STATE_TITLE:
            self.big_text("HELLBREAK", WIN_H // 3, COL_RED, size=4)
            self.med_text("A RETRO DEMON-BLASTING FPS", WIN_H // 3 + 90, COL_AMBER)
            self.med_text("ENTER - START      ESC - QUIT", WIN_H - 180)
            self.med_text("WASD move   mouse turn   LMB fire   E use   1-4 weapons",
                          WIN_H - 120, (150, 140, 130))
        elif self.state == self.STATE_PAUSE:
            self.big_text("PAUSED", WIN_H // 3, COL_AMBER, size=3)
            self.med_text("ESC - RESUME      Q - QUIT TO TITLE", WIN_H // 2)
        elif self.state == self.STATE_DEAD:
            self.win.fill((60, 8, 8))
            self.big_text("YOU DIED", WIN_H // 3, COL_RED, size=4)
            self.med_text("ENTER - RESTART LEVEL", WIN_H // 2 + 40)
        elif self.state == self.STATE_INTERMISSION:
            g = self.game
            self.big_text("LEVEL CLEARED", WIN_H // 4, COL_AMBER, size=3)
            self.med_text(g.level.name, WIN_H // 4 + 70, (150, 140, 130))
            self.med_text(f"KILLS  {g.kills} / {g.total_kills}", WIN_H // 2 - 20)
            self.med_text(f"ITEMS  {g.items_got} / {g.total_items}", WIN_H // 2 + 20)
            self.med_text(f"TIME   {int(g.time // 60)}:{int(g.time % 60):02d}",
                          WIN_H // 2 + 60)
            self.med_text("ENTER - NEXT LEVEL", WIN_H - 140, COL_AMBER)
        elif self.state == self.STATE_VICTORY:
            self.big_text("THE SPIRE HAS FALLEN", WIN_H // 3, COL_AMBER, size=2)
            self.med_text("You clawed your way through the horde.", WIN_H // 2 - 20)
            self.med_text("The breach is sealed. For now.", WIN_H // 2 + 20)
            self.med_text("ENTER - TITLE SCREEN", WIN_H - 140, COL_AMBER)

    # ------------------------------------------------------------------
    def tick_play(self, dt, events):
        g = self.game
        p = g.player
        mouse_dx = 0.0
        fire_click = False
        for ev in events:
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    self.state = self.STATE_PAUSE
                    self.grab_mouse(False)
                    return
                if ev.key in (pygame.K_e, pygame.K_SPACE):
                    p.use(g)
                if ev.key == pygame.K_TAB:
                    self.show_map = not self.show_map
                if pygame.K_1 <= ev.key <= pygame.K_4:
                    name = SLOT_TO_WEAPON[ev.key - pygame.K_0]
                    if name in p.weapons:
                        p.switch_to(name)
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                if self.mouse_grabbed:
                    fire_click = True
                else:
                    self.grab_mouse(True)
            elif ev.type == pygame.MOUSEMOTION and self.mouse_grabbed:
                mouse_dx += ev.rel[0] * MOUSE_SENS

        keys = pygame.key.get_pressed()
        keys_held = {
            "fwd": keys[pygame.K_w] or keys[pygame.K_UP],
            "back": keys[pygame.K_s] or keys[pygame.K_DOWN],
            "left": keys[pygame.K_a],
            "right": keys[pygame.K_d],
            "run": keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT],
        }
        turn = (keys[pygame.K_RIGHT] - keys[pygame.K_LEFT])
        mouse_held = pygame.mouse.get_pressed()[0] and self.mouse_grabbed
        fire_held = fire_click or mouse_held or keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]

        g.update(dt, keys_held, turn, mouse_dx, fire_held)

        if p.dead:
            self.sounds.stop_music()
            self.state = self.STATE_DEAD
            self.grab_mouse(False)
        elif g.finished:
            self.sounds.stop_music()
            if g.level_index + 1 >= len(LEVELS):
                self.state = self.STATE_VICTORY
            else:
                self.state = self.STATE_INTERMISSION
            self.grab_mouse(False)

        self.draw_play()

    def draw_play(self):
        g = self.game
        p = g.player
        frame = self.caster.render(g.level, p.x, p.y, p.angle)
        self.caster.draw_sprites(g.visible_entities(), self.assets,
                                 p.x, p.y, p.angle)
        pygame.transform.scale(frame, (WIN_W, WIN_H), self.win)

        self.hud.update(1 / 60)
        self.hud.draw_weapon(self.win, p, p.bob)
        self.hud.draw_bar(self.win, p, g)
        self.hud.draw_crosshair(self.win)
        self.hud.draw_messages(self.win, g)
        if self.show_map:
            self.hud.draw_minimap(self.win, g)

        if g.damage_flash > 0:
            overlay = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
            overlay.fill((200, 20, 10, int(120 * min(1, g.damage_flash))))
            self.win.blit(overlay, (0, 0))
        elif g.pickup_flash > 0:
            overlay = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
            overlay.fill((220, 190, 60, int(50 * g.pickup_flash / 0.25)))
            self.win.blit(overlay, (0, 0))


def main():
    App().run()


if __name__ == "__main__":
    main()
