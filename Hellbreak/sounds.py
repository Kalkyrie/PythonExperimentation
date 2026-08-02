"""Procedurally synthesized sound effects and music for HELLBREAK.

Everything is generated with numpy and fed to pygame.mixer, so no audio
files are required.  If the mixer cannot initialise (e.g. headless CI),
every sound becomes a no-op.
"""
import numpy as np
import pygame

SAMPLE_RATE = 22050
_rng = np.random.default_rng(666)


class _NullSound:
    def play(self, *a, **k):
        return None

    def set_volume(self, *a, **k):
        return None

    def stop(self):
        return None


def _env(n, attack=0.005, decay=4.0):
    t = np.linspace(0, 1, n)
    e = np.exp(-t * decay)
    a = int(attack * SAMPLE_RATE)
    if a > 0:
        e[:a] *= np.linspace(0, 1, a)
    return e


def _lowpass(x, k=6):
    kernel = np.ones(k) / k
    return np.convolve(x, kernel, mode="same")


def _tone(freq, dur, wave="square", sweep=1.0):
    n = int(dur * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    f = freq * np.linspace(1.0, sweep, n)
    phase = 2 * np.pi * np.cumsum(f) / SAMPLE_RATE
    if wave == "square":
        return np.sign(np.sin(phase))
    if wave == "saw":
        return 2 * ((phase / (2 * np.pi)) % 1.0) - 1
    if wave == "tri":
        return 2 * np.abs(2 * ((phase / (2 * np.pi)) % 1.0) - 1) - 1
    return np.sin(phase)


def _noise(dur):
    return _rng.uniform(-1, 1, int(dur * SAMPLE_RATE))


def _mix(*parts):
    n = max(len(p) for p in parts)
    out = np.zeros(n)
    for p in parts:
        out[:len(p)] += p
    m = np.abs(out).max()
    if m > 1e-9:
        out /= max(1.0, m)
    return out


def _make(arr, volume=1.0):
    init = pygame.mixer.get_init()
    if not init:
        return _NullSound()
    data = (np.clip(arr, -1, 1) * 32000).astype(np.int16)
    channels = init[2]
    if channels > 1:
        data = np.repeat(data[:, None], channels, axis=1)
    snd = pygame.sndarray.make_sound(np.ascontiguousarray(data))
    snd.set_volume(volume)
    return snd


# ---------------------------------------------------------------------------


def _pistol():
    crack = _noise(0.10) * _env(int(0.10 * SAMPLE_RATE), decay=14)
    thump = _tone(160, 0.12, "sine", sweep=0.5) * _env(int(0.12 * SAMPLE_RATE), decay=9)
    return _mix(crack, thump * 0.8)


def _shotgun():
    blast = _lowpass(_noise(0.35), 4) * _env(int(0.35 * SAMPLE_RATE), decay=7)
    boom = _tone(90, 0.3, "sine", sweep=0.4) * _env(int(0.3 * SAMPLE_RATE), decay=5)
    return _mix(blast, boom)


def _chaingun():
    crack = _noise(0.07) * _env(int(0.07 * SAMPLE_RATE), decay=18)
    thump = _tone(190, 0.07, "sine", sweep=0.6) * _env(int(0.07 * SAMPLE_RATE), decay=14)
    return _mix(crack, thump * 0.7)


def _plasma():
    zap = _tone(900, 0.16, "square", sweep=0.25) * _env(int(0.16 * SAMPLE_RATE), decay=8)
    hiss = _lowpass(_noise(0.12), 3) * _env(int(0.12 * SAMPLE_RATE), decay=10) * 0.4
    return _mix(zap, hiss)


def _explosion():
    boom = _lowpass(_noise(0.7), 10) * _env(int(0.7 * SAMPLE_RATE), decay=4)
    sub = _tone(55, 0.6, "sine", sweep=0.5) * _env(int(0.6 * SAMPLE_RATE), decay=3)
    return _mix(boom, sub)


def _fireball():
    whoosh = _lowpass(_noise(0.3), 8) * _env(int(0.3 * SAMPLE_RATE), decay=5)
    tone = _tone(300, 0.3, "saw", sweep=0.5) * _env(int(0.3 * SAMPLE_RATE), decay=5) * 0.3
    return _mix(whoosh, tone)


def _growl(freq=80):
    n = int(0.45 * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    vib = freq * (1 + 0.12 * np.sin(2 * np.pi * 9 * t))
    phase = 2 * np.pi * np.cumsum(vib) / SAMPLE_RATE
    g = np.sign(np.sin(phase)) * 0.6 + _lowpass(_noise(0.45), 12) * 0.5
    return g * _env(n, attack=0.03, decay=4)


def _death_cry():
    n = int(0.6 * SAMPLE_RATE)
    sweep = _tone(200, 0.6, "saw", sweep=0.25)
    grit = _lowpass(_noise(0.6), 8) * 0.5
    return _mix(sweep, grit) * _env(n, attack=0.01, decay=3.5)


def _pain():
    return _tone(220, 0.15, "square", sweep=0.7) * _env(int(0.15 * SAMPLE_RATE), decay=10)


def _pickup():
    a = _tone(660, 0.07, "square") * _env(int(0.07 * SAMPLE_RATE), decay=6)
    b = _tone(990, 0.09, "square") * _env(int(0.09 * SAMPLE_RATE), decay=6)
    return np.concatenate([a, b]) * 0.7


def _key_chime():
    seq = [523, 659, 784, 1047]
    parts = [_tone(f, 0.09, "tri") * _env(int(0.09 * SAMPLE_RATE), decay=4) for f in seq]
    return np.concatenate(parts) * 0.8


def _door():
    hum = _tone(60, 0.5, "square") * 0.4 + _lowpass(_noise(0.5), 14) * 0.5
    return hum * _env(int(0.5 * SAMPLE_RATE), attack=0.02, decay=2.5)


def _switch():
    click = _noise(0.03) * _env(int(0.03 * SAMPLE_RATE), decay=25)
    thunk = _tone(110, 0.18, "sine", sweep=0.6) * _env(int(0.18 * SAMPLE_RATE), decay=8)
    return _mix(click, thunk)


def _empty_click():
    return _noise(0.02) * _env(int(0.02 * SAMPLE_RATE), decay=30) * 0.5


def _music():
    """Original 8-bar doom-flavoured chug loop in E minor, ~135 bpm."""
    bpm = 135
    step = 60 / bpm / 2                      # eighth notes
    E2, G2, A2, AS2, B2, D3, E3 = 82.4, 98.0, 110.0, 116.5, 123.5, 146.8, 164.8
    riff = [E2, E2, E3, E2, E2, D3, E2, G2,
            E2, E2, E3, E2, AS2, A2, G2, E2,
            E2, E2, E3, E2, E2, D3, E2, B2,
            G2, G2, A2, AS2, B2, AS2, A2, G2]
    parts = []
    for i, f in enumerate(riff):
        n = int(step * SAMPLE_RATE)
        note = (_tone(f, step, "saw") * 0.55 + _tone(f * 1.005, step, "square") * 0.35)
        note *= _env(n, attack=0.004, decay=5.5)
        if i % 4 == 0:  # kick-ish thump
            note = _mix(note, _tone(50, min(step, 0.1), "sine", 0.5)
                        * _env(int(min(step, 0.1) * SAMPLE_RATE), decay=10))
        if i % 4 == 2:  # hat
            note = _mix(note, _noise(0.03) * _env(int(0.03 * SAMPLE_RATE), decay=30) * 0.25)
        parts.append(note)
    loop = np.concatenate(parts)
    return _lowpass(loop, 2) * 0.9


# ---------------------------------------------------------------------------


class SoundBank:
    def __init__(self):
        self.enabled = bool(pygame.mixer.get_init())
        mk = _make
        self.snd = {
            "pistol": mk(_pistol(), 0.5),
            "shotgun": mk(_shotgun(), 0.6),
            "chaingun": mk(_chaingun(), 0.4),
            "plasma": mk(_plasma(), 0.45),
            "explosion": mk(_explosion(), 0.7),
            "fireball": mk(_fireball(), 0.5),
            "growl": mk(_growl(80), 0.5),
            "growl_hi": mk(_growl(130), 0.45),
            "death": mk(_death_cry(), 0.55),
            "pain": mk(_pain(), 0.5),
            "pickup": mk(_pickup(), 0.5),
            "key": mk(_key_chime(), 0.6),
            "door": mk(_door(), 0.6),
            "switch": mk(_switch(), 0.7),
            "click": mk(_empty_click(), 0.6),
        }
        self.music = mk(_music(), 0.30)

    def play(self, name):
        s = self.snd.get(name)
        if s:
            s.play()

    def start_music(self):
        self.music.play(loops=-1)

    def stop_music(self):
        self.music.stop()
