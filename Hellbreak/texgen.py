"""Procedural 64x64 wall / floor / ceiling textures.

Everything is synthesized with numpy so the game ships with zero binary
assets.  Textures are returned as pygame Surfaces; the raycaster also wants
raw numpy arrays for the floor/ceiling mapper.
"""
import numpy as np
import pygame

from settings import TEX_SIZE

RNG = np.random.default_rng(1349)
T = TEX_SIZE


def _noise(scale=1.0, base=0.5, amp=0.5):
    """Smooth-ish value noise in [0,1], shape (T, T)."""
    small = RNG.random((T // 8, T // 8))
    big = np.kron(small, np.ones((8, 8)))
    fine = RNG.random((T, T))
    n = base + amp * (0.65 * big + 0.35 * fine - 0.5) * 2 * scale
    return np.clip(n, 0.0, 1.0)


def _to_surface(arr):
    """(T,T,3) float 0..1 -> pygame Surface."""
    a = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    surf = pygame.Surface((T, T))
    pygame.surfarray.blit_array(surf, np.transpose(a, (1, 0, 2)))
    return surf


def _colorize(gray, rgb):
    return np.stack([gray * rgb[0], gray * rgb[1], gray * rgb[2]], axis=-1)


def brick(rgb=(0.62, 0.28, 0.20), mortar=(0.22, 0.18, 0.16)):
    g = _noise(0.35, 0.75, 0.4)
    img = _colorize(g, rgb)
    bh, bw = 8, 16
    for y in range(0, T, bh):
        row = (y // bh) % 2
        img[y, :, :] = mortar
        for x in range(0, T, bw):
            xx = (x + row * bw // 2) % T
            img[y:y + bh, xx, :] = mortar
    # edge damage
    chips = RNG.random((T, T)) > 0.985
    img[chips] = np.array(mortar) * 1.3
    return img


def bigbrick(rgb=(0.45, 0.42, 0.40)):
    return brick(rgb=rgb, mortar=(0.15, 0.14, 0.13))


def tech(rgb=(0.30, 0.34, 0.38)):
    g = _noise(0.25, 0.7, 0.35)
    img = _colorize(g, rgb)
    # panel seams
    for y in (0, 21, 42, 63):
        img[y, :, :] *= 0.35
    for x in (0, 31, 32, 63):
        img[:, x, :] *= 0.35
    # rivets
    for y in (4, 17, 25, 38, 46, 59):
        for x in (4, 27, 36, 59):
            img[y - 1:y + 1, x - 1:x + 1, :] = (0.75, 0.78, 0.8)
    # glowing status lights
    for i, x in enumerate(range(8, 56, 10)):
        col = (0.1, 0.9, 0.2) if i % 3 else (0.9, 0.2, 0.1)
        img[10:12, x:x + 3, :] = col
    return img


def rust(rgb=(0.45, 0.30, 0.18)):
    g = _noise(0.6, 0.55, 0.5)
    img = _colorize(g, rgb)
    streaks = np.cumsum(RNG.random((T, T)) - 0.5, axis=0)
    streaks = (streaks - streaks.min()) / (np.ptp(streaks) + 1e-6)
    img *= (0.7 + 0.5 * streaks)[..., None]
    for x in (0, 63):
        img[:, x, :] *= 0.4
    return np.clip(img, 0, 1)


def bone(rgb=(0.72, 0.66, 0.52)):
    """Hellish wall of stacked bone-like arches (original design)."""
    g = _noise(0.3, 0.6, 0.35)
    img = _colorize(g, rgb)
    yy, xx = np.mgrid[0:T, 0:T]
    for cy in range(8, T, 16):
        for cx in range(8, T, 16):
            d = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
            ring = np.abs(d - 5.5) < 1.6
            img[ring] = np.array(rgb) * 0.35          # eye sockets / hollows
            dome = d < 4.0
            img[dome] *= 1.25
    img[:, ::16, :] *= 0.5
    return np.clip(img, 0, 1)


def hellrock(rgb=(0.42, 0.16, 0.12)):
    g = _noise(0.7, 0.5, 0.5)
    img = _colorize(g, rgb)
    # glowing lava cracks
    crack = np.zeros((T, T), bool)
    x = RNG.integers(0, T)
    for y in range(T):
        x = (x + RNG.integers(-1, 2)) % T
        crack[y, x] = True
        if RNG.random() < 0.2:
            crack[y, (x + 1) % T] = True
    glow = np.zeros((T, T))
    glow[crack] = 1.0
    k = np.array([0.25, 0.5, 1.0, 0.5, 0.25])
    for axis in (0, 1):
        glow = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), axis, glow)
    img[..., 0] += glow * 0.9
    img[..., 1] += glow * 0.35
    return np.clip(img, 0, 1)


def door_tex(trim=(0.55, 0.55, 0.58)):
    g = _noise(0.2, 0.62, 0.3)
    img = _colorize(g, (0.35, 0.36, 0.40))
    # horizontal slats
    for y in range(6, T - 6, 7):
        img[y, 4:T - 4, :] *= 0.45
    # frame
    img[:4, :, :] = trim
    img[-4:, :, :] = trim
    img[:, :3, :] = np.array(trim) * 0.8
    img[:, -3:, :] = np.array(trim) * 0.8
    # center handle plate
    img[28:36, 28:36, :] = (0.7, 0.65, 0.3)
    img[30:34, 30:34, :] = (0.4, 0.35, 0.15)
    return img


def key_door(col):
    img = door_tex()
    stripe = {"red": (0.85, 0.12, 0.10), "blue": (0.15, 0.35, 0.95)}[col]
    img[6:12, 6:T - 6, :] = stripe
    img[T - 12:T - 6, 6:T - 6, :] = stripe
    return img


def exit_switch(pressed=False):
    g = _noise(0.2, 0.6, 0.3)
    img = _colorize(g, (0.32, 0.30, 0.28))
    img[:3, :, :] = (0.5, 0.5, 0.5)
    img[-3:, :, :] = (0.5, 0.5, 0.5)
    # switch housing
    img[16:48, 20:44, :] = (0.15, 0.15, 0.15)
    lever = (0.9, 0.15, 0.1) if not pressed else (0.1, 0.9, 0.15)
    if pressed:
        img[36:46, 26:38, :] = lever
    else:
        img[18:28, 26:38, :] = lever
    # EXIT letters (blocky, drawn as bright pixels)
    letters = ["1110101011101", "1000101001000", "1100010001000",
               "1000101001000", "1110101011000"]
    for dy, row in enumerate(letters):
        for dx, c in enumerate(row):
            if c == "1":
                img[52 + dy, 25 + dx, :] = (0.95, 0.8, 0.2)
    return img


def crate():
    g = _noise(0.3, 0.6, 0.3)
    img = _colorize(g, (0.55, 0.42, 0.22))
    for i in (0, 1, 62, 63):
        img[i, :, :] *= 0.4
        img[:, i, :] *= 0.4
    yy, xx = np.mgrid[0:T, 0:T]
    diag = np.abs(yy - xx) < 2
    diag2 = np.abs(yy + xx - T) < 2
    img[diag] *= 0.55
    img[diag2] *= 0.55
    return img


def floor_concrete():
    g = _noise(0.35, 0.5, 0.35)
    img = _colorize(g, (0.34, 0.33, 0.31))
    img[::32, :, :] *= 0.6
    img[:, ::32, :] *= 0.6
    return img


def floor_hex():
    g = _noise(0.25, 0.55, 0.3)
    img = _colorize(g, (0.30, 0.26, 0.22))
    for y in range(0, T, 16):
        img[y, :, :] *= 0.55
    for x in range(0, T, 16):
        img[:, x, :] *= 0.55
    return img


def floor_blood():
    g = _noise(0.6, 0.45, 0.5)
    img = _colorize(g, (0.40, 0.14, 0.10))
    pools = _noise(0.9, 0.35, 0.6) > 0.62
    img[pools] = (0.45, 0.03, 0.02)
    return img


def ceil_metal():
    g = _noise(0.25, 0.4, 0.25)
    img = _colorize(g, (0.22, 0.23, 0.26))
    img[::16, :, :] *= 0.5
    img[:, ::16, :] *= 0.5
    for y in range(8, T, 16):
        for x in range(8, T, 16):
            img[y - 1:y + 2, x - 1:x + 2, :] = (0.55, 0.55, 0.4)
    return img


def ceil_cave():
    g = _noise(0.8, 0.35, 0.45)
    return _colorize(g, (0.30, 0.16, 0.12))


# ---------------------------------------------------------------------------
# Texture registry.  Wall ids are the positive ints used in the level grid.
# ---------------------------------------------------------------------------
WALL_BUILDERS = {
    1: lambda: brick(),                       # red brick
    2: tech,                                  # base tech wall
    3: bigbrick,                              # grey stone
    4: rust,                                  # rusted metal
    5: bone,                                  # bone wall
    6: hellrock,                              # glowing hell rock
    7: door_tex,                              # plain door
    8: lambda: key_door("blue"),
    9: lambda: key_door("red"),
    10: exit_switch,                          # exit switch (unpressed)
    11: lambda: exit_switch(True),            # exit switch (pressed)
    12: crate,
}

FLOOR_BUILDERS = {
    "concrete": floor_concrete,
    "hex": floor_hex,
    "blood": floor_blood,
}
CEIL_BUILDERS = {
    "metal": ceil_metal,
    "cave": ceil_cave,
}


def build_all():
    """Returns (wall_surfaces, wall_columns, floor_arrays, ceil_arrays).

    wall_surfaces: id -> (bright Surface, dark Surface)
    wall_columns:  id -> (bright col list, dark col list) 1px wide subsurfaces
    floor/ceil arrays: name -> (T,T,3) uint8 numpy array
    """
    walls, wall_cols = {}, {}
    for tid, fn in WALL_BUILDERS.items():
        arr = fn()
        bright = _to_surface(arr)
        dark = _to_surface(arr * 0.72)
        walls[tid] = (bright, dark)
        wall_cols[tid] = (
            [bright.subsurface(x, 0, 1, T) for x in range(T)],
            [dark.subsurface(x, 0, 1, T) for x in range(T)],
        )
    floors = {n: (np.clip(fn(), 0, 1) * 255).astype(np.uint8)
              for n, fn in FLOOR_BUILDERS.items()}
    ceils = {n: (np.clip(fn(), 0, 1) * 255).astype(np.uint8)
             for n, fn in CEIL_BUILDERS.items()}
    return walls, wall_cols, floors, ceils
