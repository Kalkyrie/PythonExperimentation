"""Global constants and tuning values for HELLBREAK."""

# --- Display ---
RENDER_W, RENDER_H = 320, 200          # internal (retro) resolution
SCALE = 3                              # window = internal * SCALE
WIN_W, WIN_H = RENDER_W * SCALE, RENDER_H * SCALE
FPS_CAP = 60

HALF_H = RENDER_H // 2
TEX_SIZE = 64

# --- Camera ---
PLANE_LEN = 0.66                       # ~66 degree FOV
MOUSE_SENS = 0.0022
TURN_SPEED = 2.6                       # radians/sec (keyboard)

# --- Player ---
PLAYER_RADIUS = 0.28
MOVE_SPEED = 3.4
RUN_MULT = 1.55
PLAYER_MAX_HEALTH = 100
PLAYER_MAX_ARMOR = 100
USE_RANGE = 1.1

# --- Doors ---
DOOR_SPEED = 1.6                       # openness units per second
DOOR_OPEN_TIME = 4.0                   # seconds before auto-close
DOOR_PASSABLE = 0.75

# --- Rendering / lighting ---
FOG_STRENGTH = 0.045                   # distance attenuation
MIN_BRIGHT = 0.14
SIDE_SHADE = 0.72                      # darkening of N/S facing walls
MAX_SPRITE_H = RENDER_H * 4

# --- Ammo ---
AMMO_MAX = {"bullets": 200, "shells": 50, "cells": 200}

# --- Colors ---
COL_HUD_BG = (28, 24, 22)
COL_HUD_EDGE = (70, 60, 55)
COL_AMBER = (232, 174, 62)
COL_RED = (200, 40, 30)
COL_GREEN = (70, 200, 70)
COL_BLUE = (70, 120, 230)
COL_TEXT = (215, 205, 190)
