"""All sprite artwork for HELLBREAK.

Sprites are authored as ASCII pixel maps (one char per pixel, per-sprite
palette) and turned into pygame Surfaces at load time.  Effects such as
fireballs, muzzle flashes and explosions are synthesized with radial
gradients.  Pain / death animation frames are derived automatically
(red tint, squash-and-collapse), which keeps the hand-drawn set small.
"""
import math

import pygame

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def build(rows, palette, scale=1):
    """ASCII rows -> Surface with per-pixel alpha ('.' = transparent)."""
    h = len(rows)
    w = max(len(r) for r in rows)
    surf = pygame.Surface((w, h), pygame.SRCALPHA)
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch != ".":
                surf.set_at((x, y), palette[ch])
    if scale != 1:
        surf = pygame.transform.scale(surf, (w * scale, h * scale))
    return surf


def tint(surf, rgb, amount=110):
    s = surf.copy()
    overlay = pygame.Surface(s.get_size(), pygame.SRCALPHA)
    overlay.fill((*rgb, 0))
    mask = pygame.surfarray.pixels_alpha(s)
    import numpy as np
    add = pygame.Surface(s.get_size(), pygame.SRCALPHA)
    arr = pygame.surfarray.pixels3d(add)
    arr[:, :, 0] = rgb[0]
    arr[:, :, 1] = rgb[1]
    arr[:, :, 2] = rgb[2]
    aarr = pygame.surfarray.pixels_alpha(add)
    aarr[:, :] = (mask > 0) * amount
    del arr, aarr, mask
    s.blit(add, (0, 0))
    return s


def squash(surf, factor):
    """Collapse sprite toward the ground (for death animations)."""
    w, h = surf.get_size()
    nh = max(2, int(h * factor))
    sq = pygame.transform.scale(surf, (int(w * (1 + (1 - factor) * 0.5)), nh))
    out = pygame.Surface((sq.get_width(), h), pygame.SRCALPHA)
    out.blit(sq, (0, h - nh))
    return out


def gib_pile(w, h, base=(120, 20, 15)):
    surf = pygame.Surface((w, h), pygame.SRCALPHA)
    import random
    rnd = random.Random(w * 7 + h)
    for _ in range(w // 2):
        x = rnd.randint(2, w - 4)
        y = rnd.randint(h - max(3, h // 6), h - 1)
        r = rnd.randint(1, 3)
        c = (min(255, base[0] + rnd.randint(-30, 60)),
             max(0, base[1] + rnd.randint(-10, 25)),
             base[2])
        pygame.draw.circle(surf, c, (x, y), r)
    for _ in range(4):
        x = rnd.randint(4, w - 6)
        y = h - rnd.randint(2, max(3, h // 8))
        pygame.draw.rect(surf, (200, 190, 170), (x, y, 3, 2))  # bone bits
    return surf


def glow_ball(radius, inner, outer, spikes=0):
    d = radius * 2
    surf = pygame.Surface((d, d), pygame.SRCALPHA)
    for r in range(radius, 0, -1):
        t = r / radius
        col = [int(inner[i] * (1 - t) + outer[i] * t) for i in range(3)]
        alpha = int(255 * (1.0 - t * 0.75))
        pygame.draw.circle(surf, (*col, alpha), (radius, radius), r)
    if spikes:
        for i in range(spikes):
            a = i * (2 * math.pi / spikes)
            x = radius + int(math.cos(a) * radius * 0.95)
            y = radius + int(math.sin(a) * radius * 0.95)
            pygame.draw.line(surf, (*inner, 200), (radius, radius), (x, y), 2)
    return surf


def explosion_frames(size=64, n=6, hot=(255, 240, 160), mid=(255, 120, 30)):
    frames = []
    for i in range(n):
        t = (i + 1) / n
        surf = pygame.Surface((size, size), pygame.SRCALPHA)
        r = int(size * 0.5 * (0.3 + 0.7 * t))
        cx = cy = size // 2
        import random
        rnd = random.Random(i * 31)
        for _ in range(10 + i * 4):
            a = rnd.random() * math.tau
            dist = rnd.random() * r
            br = max(2, int((1 - t) * size * 0.14 * rnd.uniform(0.5, 1.4)))
            px = cx + int(math.cos(a) * dist)
            py = cy + int(math.sin(a) * dist * 0.85)
            frac = dist / (r + 1)
            col = hot if frac < 0.4 and t < 0.6 else mid
            if t > 0.7:
                col = (90, 80, 75)  # smoke
            alpha = int(230 * (1 - t * 0.8))
            pygame.draw.circle(surf, (*col, alpha), (px, py), br)
        frames.append(surf)
    return frames


# ---------------------------------------------------------------------------
# ENEMY: "GRUNT" — possessed soldier with a rifle (hitscan)
# ---------------------------------------------------------------------------
GRUNT_PAL = {
    "K": (18, 14, 14), "s": (140, 145, 118), "S": (170, 175, 145),
    "g": (70, 92, 50), "d": (44, 60, 32), "m": (55, 55, 60),
    "M": (95, 95, 105), "r": (150, 25, 20), "e": (240, 60, 40),
    "b": (35, 30, 26), "y": (200, 170, 60),
}
GRUNT_WALK = [
    "......KKKKKK........",
    ".....KssssssK.......",
    ".....KsSssSsK.......",
    ".....KseKKesK.......",
    ".....KssssssK.......",
    "......KsrrsK........",
    ".....KKKssKKK.......",
    "...KKgggggggdKK.....",
    "..KgggggggggddK.....",
    ".KgKgggggggdKdK.....",
    ".KgKgggdggddKdK.....",
    ".KsKggdddgddKsK.....",
    ".KsKMmmmmmmmKsK.....",
    ".KKKmMMMMMmmKKK.....",
    "....KggggddK........",
    "....KggggddK........",
    "....KggKKddK........",
    "....KgKKKKdK........",
    "....KgK..KdK........",
    "...KggK..KddK.......",
    "...KggK..KddK.......",
    "...KbbK..KbbK.......",
    "..KbbbK..KbbbK......",
    "..KKKKK..KKKKK......",
]
GRUNT_FIRE = [
    "......KKKKKK........",
    ".....KssssssK.......",
    ".....KsSssSsK.......",
    ".....KseKKesK.......",
    ".....KssssssK.......",
    "......KsrrsK........",
    ".....KKKssKKK.......",
    "...KKgggggggdKK.....",
    "..KgggggggggddKK....",
    ".KgKgggggggddKssKK..",
    ".KgKgggdggddKMmmmmMM",
    ".KsKggdddgddKMmmmmMM",
    ".KsKgggggggdKssKK...",
    ".KKKggggggddKKK.....",
    "....KggggddK........",
    "....KggggddK........",
    "....KggKKddK........",
    "....KgKKKKdK........",
    "....KgK..KdK........",
    "...KggK..KddK.......",
    "...KggK..KddK.......",
    "...KbbK..KbbK.......",
    "..KbbbK..KbbbK......",
    "..KKKKK..KKKKK......",
]

# ---------------------------------------------------------------------------
# ENEMY: "FIEND" — horned demon, throws fireballs
# ---------------------------------------------------------------------------
FIEND_PAL = {
    "K": (20, 12, 10), "b": (140, 84, 48), "B": (170, 110, 62),
    "d": (95, 55, 30), "h": (225, 215, 190), "e": (255, 170, 30),
    "c": (60, 34, 20), "r": (170, 30, 20), "f": (255, 120, 20),
    "F": (255, 220, 90),
}
FIEND_WALK = [
    "..Kh..........hK....",
    ".KhhK........KhhK...",
    ".KhKK..KKKK..KKhK...",
    ".KhK..KbbbbK..KhK...",
    "..K..KbBbbBbK..K....",
    ".....KbeKKebK.......",
    ".....KbbbbbbK.......",
    "......KrrrrK........",
    ".....KKbbbbKK.......",
    "...KKbbBbbBbbKK.....",
    "..KbbKbbbbbbKbbK....",
    ".KbbK.KbbbbK.KbbK...",
    ".KbK..KbBBbK..KbK...",
    ".KbK..KbbbbK..KbK...",
    ".KcK..KbbbbK..KcK...",
    "..K...KbddbK...K....",
    "......KbKKbK........",
    ".....KbbK.KbbK......",
    ".....KbK...KbK......",
    "....KbbK...KbbK.....",
    "....KcK.....KcK.....",
    "...KccK.....KccK....",
    "...KKKK.....KKKK....",
]
FIEND_ATTACK = [
    "..Kh..........hK....",
    ".KhhK........KhhK...",
    ".KhKK..KKKK..KKhK.FF",
    ".KhK..KbbbbK..KhKFff",
    "..K..KbBbbBbK.KbKFff",
    ".....KbeKKebK.KbK.FF",
    ".....KbbbbbbK.KbK...",
    "......KrrrrKKKbK....",
    ".....KKbbbbKbbK.....",
    "...KKbbBbbBbbKK.....",
    "..KbbKbbbbbbKK......",
    ".KbbK.KbbbbK........",
    ".KbK..KbBBbK........",
    ".KbK..KbbbbK........",
    ".KcK..KbbbbK........",
    "..K...KbddbK........",
    "......KbKKbK........",
    ".....KbbK.KbbK......",
    ".....KbK...KbK......",
    "....KbbK...KbbK.....",
    "....KcK.....KcK.....",
    "...KccK.....KccK....",
    "...KKKK.....KKKK....",
]

# ---------------------------------------------------------------------------
# ENEMY: "RAVAGER" — hulking pink bruiser, melee only
# ---------------------------------------------------------------------------
RAV_PAL = {
    "K": (24, 10, 12), "p": (190, 90, 95), "P": (220, 130, 130),
    "d": (140, 55, 60), "t": (240, 235, 210), "e": (60, 220, 70),
    "m": (120, 40, 45), "r": (120, 15, 15),
}
RAV_WALK = [
    ".......KKKKKKKK.........",
    ".....KKppppppppKK.......",
    "....KppppPPPPppppK......",
    "...KpppPPPPPPPPpppK.....",
    "...KppKeKPPPPKeKppK.....",
    "...KppppPPPPPPppppK.....",
    "...KpKKKKKKKKKKKKpK.....",
    "...KpKtKtKtKtKtKKpK.....",
    "...KppKrrrrrrrrKppK.....",
    "..KppppKKKKKKKKppppK....",
    ".KppppppppppppppppppK...",
    ".KpppKppppppppppKpppK...",
    ".KppK.KppPPPPppK.KppK...",
    ".KppK.KppppppppK.KppK...",
    ".KpdK.KppppppppK.KpdK...",
    ".KKK..KdppppppdK..KKK...",
    "......KdpKKKKpdK........",
    ".....KdppK..KppdK.......",
    ".....KdpK....KpdK.......",
    "....KddpK....KpddK......",
    "....KmmK......KmmK......",
    "...KmmmK......KmmmK.....",
    "...KKKKK......KKKKK.....",
]
RAV_ATTACK = [
    ".......KKKKKKKK.........",
    ".....KKppppppppKK.......",
    "....KppppPPPPppppK......",
    "...KpppPPPPPPPPpppK.....",
    "...KppKeKPPPPKeKppK.....",
    "...KppppPPPPPPppppK.....",
    "...KpKKKKKKKKKKKKpK.....",
    "...KpKtKtKtKtKtKKpK.....",
    "...KppKrrrrrrrrKppK.....",
    "KKpppppKKKKKKKKpppppKK..",
    "KppppppppppppppppppppK..",
    "KppKKppppppppppppKKppK..",
    "KppK.KppPPPPppK...KppK..",
    "KttK..KppppppppK..KttK..",
    ".KK...KppppppppK...KK...",
    "......KdppppppdK........",
    "......KdpKKKKpdK........",
    ".....KdppK..KppdK.......",
    ".....KdpK....KpdK.......",
    "....KddpK....KpddK......",
    "....KmmK......KmmK......",
    "...KmmmK......KmmmK.....",
    "...KKKKK......KKKKK.....",
]

# ---------------------------------------------------------------------------
# Pickups
# ---------------------------------------------------------------------------
MED_PAL = {"K": (20, 20, 20), "w": (225, 225, 220), "g": (30, 180, 60),
           "s": (160, 160, 155)}
MEDKIT = [
    "KKKKKKKKKKKKKK",
    "KwwwwwwwwwwwwK",
    "KwwwwwggwwwwwK",
    "KwwwwwggwwwwwK",
    "KwwwggggggwwwK",
    "KwwwggggggwwwK",
    "KwwwwwggwwwwwK",
    "KwwwwwggwwwwwK",
    "KssssssssssssK",
    "KKKKKKKKKKKKKK",
]
STIM = [
    "..KKKKKK..",
    ".KwwwwwwK.",
    ".KwwggwwK.",
    ".KwggggwK.",
    ".KwwggwwK.",
    ".KssssssK.",
    ".KKKKKKKK.",
]
AMMO_PAL = {"K": (20, 18, 14), "y": (200, 170, 60), "d": (140, 115, 40),
            "b": (180, 120, 30), "g": (90, 90, 90)}
CLIP = [
    "...KKKK...",
    "..KbbbbK..",
    "..KbbbbK..",
    ".KyyyyyyK.",
    ".KyddyddK.",
    ".KyyyyyyK.",
    ".KKKKKKKK.",
]
SHELLBOX_PAL = {"K": (20, 14, 12), "r": (170, 45, 30), "R": (210, 80, 50),
                "y": (220, 190, 90), "d": (120, 30, 20)}
SHELLBOX = [
    "KKKKKKKKKKKKKK",
    "KrrrrrrrrrrrrK",
    "KrRRRRRRRRRRrK",
    "KryKyKyKyKyKrK",
    "KryKyKyKyKyKrK",
    "KrddddddddddrK",
    "KKKKKKKKKKKKKK",
]
CELL_PAL = {"K": (14, 16, 24), "b": (60, 110, 220), "B": (130, 180, 255),
            "w": (220, 230, 255), "d": (30, 55, 130)}
CELLPACK = [
    "..KKKKKKKK..",
    ".KbbbbbbbbK.",
    ".KbBBBBBBbK.",
    ".KbBwwwwBbK.",
    ".KbBwBBwBbK.",
    ".KbBwwwwBbK.",
    ".KbBBBBBBbK.",
    ".KddddddddK.",
    ".KKKKKKKKKK.",
]
ARMOR_PAL = {"K": (16, 20, 16), "g": (60, 160, 70), "G": (110, 210, 110),
             "d": (35, 95, 45)}
ARMOR = [
    "..KKK..KKK..",
    ".KgggKKgggK.",
    ".KggggggggK.",
    ".KgGGggGGgK.",
    ".KgGGggGGgK.",
    ".KggggggggK.",
    "..KggddggK..",
    "..KgddddgK..",
    "...KKKKKK...",
]
KEY_PAL_BLUE = {"K": (12, 14, 26), "c": (70, 120, 240), "C": (150, 190, 255),
                "w": (230, 235, 245)}
KEY_PAL_RED = {"K": (26, 12, 12), "c": (230, 60, 40), "C": (255, 140, 110),
               "w": (245, 235, 230)}
KEYCARD = [
    "..KKKKKKK..",
    ".KcccccccK.",
    ".KcCCCCCcK.",
    ".KcCwwwCcK.",
    ".KcCCCCCcK.",
    ".KcccccccK.",
    ".KcccccccK.",
    ".KKKKKKKKK.",
]
BARREL_PAL = {"K": (14, 18, 12), "g": (60, 90, 55), "G": (95, 130, 80),
              "t": (120, 230, 60), "T": (200, 255, 120), "d": (35, 55, 32)}
BARREL = [
    "...KKKKKKKK...",
    "..KGGGGGGGGK..",
    ".KgGggggggGgK.",
    ".KggggggggggK.",
    ".KKKKKKKKKKKK.",
    ".KggttttttggK.",
    ".KgtTTttTTtgK.",
    ".KgttttttttgK.",
    ".KgtTttttTtgK.",
    ".KggttttttggK.",
    ".KKKKKKKKKKKK.",
    ".KddddddddddK.",
    ".KgddddddddgK.",
    "..KKKKKKKKKK..",
]
WEAPON_PICKUP_PAL = {"K": (16, 16, 16), "m": (90, 90, 100), "M": (140, 140, 150),
                     "w": (110, 75, 40), "y": (200, 170, 60)}
SHOTGUN_PICKUP = [
    "....................",
    "KKKK................",
    "KwwKKKKKKKKKKKKKKKK.",
    "KwwwmmmmmmmmmmmmmmMK",
    ".KwwmMMMMMMMMMMMMMMK",
    "..KKwwKKKKKKKKKKKKK.",
    "....KwwK............",
    ".....KKK............",
]
CHAINGUN_PICKUP = [
    "....................",
    ".KKKKKKKKKKKKKKKK...",
    "KmMMMMMMMMMMMMMMmK..",
    "KmMKKMKKMKKMKKMKmKK.",
    "KmMMMMMMMMMMMMMMmMK.",
    "KmMKKMKKMKKMKKMKmKK.",
    "KmMMMMMMMMMMMMMMmK..",
    ".KKKKwwKKKKKKKKKK...",
    "....KwwK............",
    ".....KKK............",
]
PLASMA_PICKUP = [
    "....................",
    ".KKKKKKKKKKKKKK.....",
    "KmmmmmmmmmmmmmmKKK..",
    "KmBBBBBBBBBBBBmmmMK.",
    "KmBKKBKKBKKBBBmmmMK.",
    "KmmmmmmmmmmmmmmKKK..",
    ".KKwwKKKKKKKKKK.....",
    "...KwwK.............",
    "....KKK.............",
]
PLASMA_PICKUP_PAL = {"K": (14, 14, 20), "m": (80, 85, 100), "M": (130, 140, 160),
                     "B": (90, 150, 255), "w": (110, 75, 40)}

# ---------------------------------------------------------------------------
# First-person weapon sprites (drawn low-res, scaled up in-game)
# ---------------------------------------------------------------------------
WPN_PAL = {
    "K": (12, 12, 14), "m": (70, 72, 80), "M": (110, 115, 125),
    "L": (150, 155, 168), "g": (150, 110, 70), "G": (180, 140, 95),
    "w": (95, 65, 40), "y": (210, 180, 70), "B": (90, 150, 255),
    "b": (50, 90, 180),
}
PISTOL = [
    "........KKKK........",
    ".......KMMMMK.......",
    ".......KMLLMK.......",
    ".......KMLLMK.......",
    ".......KMLLMK.......",
    "......KmMLLMmK......",
    "......KmMMMMmK......",
    "......KmmmmmmK......",
    ".....KgGGGGGGgK.....",
    ".....KgGggggGgK.....",
    "....KgGGGGGGGGgK....",
    "....KgGggggggGgK....",
    "...KgGGGGGGGGGGgK...",
    "...KggggggggggggK...",
]
SHOTGUN_VIEW = [
    "........KKKKKK........",
    ".......KmMMMMmK.......",
    ".......KMLLLLMK.......",
    ".......KMLKKLMK.......",
    ".......KMLKKLMK.......",
    ".......KMLLLLMK.......",
    ".......KMLLLLMK.......",
    "......KmMLLLLMmK......",
    "......KmMMMMMMmK......",
    "......KwgGGGGgwK......",
    ".....KwgGGGGGGgwK.....",
    ".....KwgGggggGgwK.....",
    "....KwgGGGGGGGGgwK....",
    "....KwgGggggggGgwK....",
    "...KwgGGGGGGGGGGgwK...",
    "...KwwwwwwwwwwwwwwK...",
]
CHAINGUN_VIEW = [
    "...KKKK..KKKK..KKKK...",
    "..KMLLMKKMLLMKKMLLMK..",
    "..KMKKMKKMKKMKKMKKMK..",
    "..KMLLMKKMLLMKKMLLMK..",
    "..KMLLMKKMLLMKKMLLMK..",
    ".KKMMMMKKMMMMKKMMMMKK.",
    ".KmmmmmmmmmmmmmmmmmmK.",
    ".KmMMMMMMMMMMMMMMMMmK.",
    ".KmmmmmmmmmmmmmmmmmmK.",
    "..KgGGGGGGGGGGGGGGgK..",
    "..KgGggggggggggggGgK..",
    "...KgGGGGGGGGGGGGgK...",
    "...KggggggggggggggK...",
]
PLASMA_VIEW = [
    ".......KKKKKKKK.......",
    "......KmMMMMMMmK......",
    "......KMBBBBBBMK......",
    "......KMBbbbbBMK......",
    "......KMBbBBbBMK......",
    "......KMBbbbbBMK......",
    "......KMBBBBBBMK......",
    ".....KmMMMMMMMMmK.....",
    ".....KmmmmmmmmmmK.....",
    ".....KgGGGGGGGGgK.....",
    "....KgGggggggggGgK....",
    "....KgGGGGGGGGGGgK....",
    "...KggggggggggggggK...",
]

# ---------------------------------------------------------------------------
# HUD face (marine helmet, 3 damage states + grin)
# ---------------------------------------------------------------------------
FACE_PAL = {
    "K": (16, 14, 12), "s": (196, 150, 115), "S": (222, 178, 140),
    "h": (70, 92, 50), "H": (95, 120, 70), "e": (240, 240, 235),
    "p": (30, 60, 150), "r": (170, 30, 25), "d": (120, 90, 70),
    "b": (150, 110, 85),
}
FACE_OK = [
    "...KKKKKKKKKK...",
    "..KhhhhhhhhhhK..",
    ".KhHHHHHHHHHHhK.",
    ".KhhhhhhhhhhhhK.",
    ".KssSSssssSSssK.",
    ".KsKeKssssKeKsK.",
    ".KsKpKssssKpKsK.",
    ".KssssssssssssK.",
    ".KsssSSssSSsssK.",
    ".KssssssssssssK.",
    ".KssKKKKKKKKssK.",
    ".KsssssssssssK..",
    "..KsssssssssK...",
    "...KKKKKKKKK....",
]
FACE_HURT = [
    "...KKKKKKKKKK...",
    "..KhhhhhhhhhhK..",
    ".KhHHHHHHHHHHhK.",
    ".KhhhhhhhhhhhhK.",
    ".KssSSssrsSSssK.",
    ".KsKeKssrsKeKsK.",
    ".KsKpKssrsKpKsK.",
    ".KssssssrsssssK.",
    ".KsssSSssSSsssK.",
    ".KssrrssssssssK.",
    ".KssKKKKKKKKssK.",
    ".KsssKKKKKKssK..",
    "..KsssssssssK...",
    "...KKKKKKKKK....",
]
FACE_BAD = [
    "...KKKKKKKKKK...",
    "..KhhhhhhhhhhK..",
    ".KhHHHHrHHHHhhK.",
    ".KhhhhrrrhhhhhK.",
    ".KsrsSrrssSSrsK.",
    ".KsKKKsrrsKKKsK.",
    ".KsKKKsrrsKKKsK.",
    ".KsrssssrsrsssK.",
    ".KsssSSrsSSrssK.",
    ".KsrrrssssrrssK.",
    ".KrsKKKKKKKKsrK.",
    ".KssrKKKKKKrsK..",
    "..KsssrrrsssK...",
    "...KKKKKKKKK....",
]

# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def build_enemy_frames(walk_rows, attack_rows, palette, scale, gib_base):
    walk_a = build(walk_rows, palette, scale)
    walk_b = pygame.transform.flip(walk_a, True, False)
    attack = build(attack_rows, palette, scale)
    pain = tint(walk_a, (255, 40, 30), 120)
    w, h = walk_a.get_size()
    death = [
        tint(squash(walk_a, 0.75), (255, 60, 40), 70),
        tint(squash(walk_a, 0.45), (200, 40, 30), 110),
        squash(walk_a, 0.22),
        gib_pile(w, h, gib_base),
    ]
    return {
        "walk": [walk_a, walk_b],
        "attack": [attack],
        "pain": [pain],
        "death": death,
        "corpse": death[-1],
    }


def build_all():
    """Build every sprite surface. Requires pygame.display to be initialised."""
    S = {}
    S["grunt"] = build_enemy_frames(GRUNT_WALK, GRUNT_FIRE, GRUNT_PAL, 3, (90, 25, 18))
    S["fiend"] = build_enemy_frames(FIEND_WALK, FIEND_ATTACK, FIEND_PAL, 3, (120, 40, 15))
    S["ravager"] = build_enemy_frames(RAV_WALK, RAV_ATTACK, RAV_PAL, 4, (150, 40, 45))

    S["medkit"] = build(MEDKIT, MED_PAL, 2)
    S["stim"] = build(STIM, MED_PAL, 2)
    S["clip"] = build(CLIP, AMMO_PAL, 2)
    S["shellbox"] = build(SHELLBOX, SHELLBOX_PAL, 2)
    S["cellpack"] = build(CELLPACK, CELL_PAL, 2)
    S["armor"] = build(ARMOR, ARMOR_PAL, 2)
    S["key_blue"] = build(KEYCARD, KEY_PAL_BLUE, 2)
    S["key_red"] = build(KEYCARD, KEY_PAL_RED, 2)
    S["barrel"] = build(BARREL, BARREL_PAL, 3)
    S["wp_shotgun"] = build(SHOTGUN_PICKUP, WEAPON_PICKUP_PAL, 2)
    S["wp_chaingun"] = build(CHAINGUN_PICKUP, WEAPON_PICKUP_PAL, 2)
    S["wp_plasma"] = build(PLASMA_PICKUP, PLASMA_PICKUP_PAL, 2)

    S["fireball"] = [glow_ball(14, (255, 240, 150), (200, 60, 10), 6),
                     glow_ball(12, (255, 220, 120), (220, 80, 10), 5)]
    S["plasmaball"] = [glow_ball(10, (220, 240, 255), (40, 90, 230), 4),
                       glow_ball(9, (200, 230, 255), (60, 110, 250), 4)]
    S["explosion"] = explosion_frames(96, 6)
    S["small_explosion"] = explosion_frames(48, 5)
    S["muzzle"] = glow_ball(16, (255, 255, 200), (255, 150, 30), 8)

    S["view_pistol"] = build(PISTOL, WPN_PAL, 10)
    S["view_shotgun"] = build(SHOTGUN_VIEW, WPN_PAL, 10)
    S["view_chaingun"] = build(CHAINGUN_VIEW, WPN_PAL, 10)
    S["view_plasma"] = build(PLASMA_VIEW, WPN_PAL, 10)

    S["face_ok"] = build(FACE_OK, FACE_PAL, 3)
    S["face_hurt"] = build(FACE_HURT, FACE_PAL, 3)
    S["face_bad"] = build(FACE_BAD, FACE_PAL, 3)
    return S
