"""Typed scene: board config, measured objects, and a layout version that plans are pinned to."""
import json
import math
import os
from dataclasses import asdict, dataclass, field


@dataclass
class Config:
    # Board frame: millimetres, origin at the top-left corner, +x right, +y downhill.
    board_w: float = 600.0
    board_h: float = 450.0
    tilt_deg: float = 8.0  # measured with a phone inclinometer; supplied, not estimated
    ball_r: float = 20.0  # 40 mm ping-pong ball
    marker_mm: float = 40.0  # printed ArUco side length
    # Board-corner marker centres (DICT_4X4_50 ids 0..3, upright on the board).
    markers: dict = field(default_factory=lambda: {0: (30, 30), 1: (570, 30), 2: (570, 420), 3: (30, 420)})
    target_id: int = 4  # the user places this marker where the ball should go
    target_r: float = 22.0  # hit = ball centre passes within this distance of the target centre
    release: tuple = (300.0, 45.0)  # fixed release gate
    guide_len: float = 180.0
    guide_w: float = 15.0
    guide_region: tuple = (70.0, 110.0, 530.0, 345.0)  # x0, y0, x1, y1 where the guide may be placed
    # HSV ranges (OpenCV H is 0..180). ponytail: the calibration knob; tune per room lighting.
    hsv: dict = field(default_factory=lambda: {
        "ball": [[5, 150, 120], [22, 255, 255]],
        "guide": [[100, 120, 50], [125, 255, 255]],
        "obstacle": [[145, 100, 60], [172, 255, 255]],
    })
    min_markers: int = 3  # fewer visible board markers = uncertain view

    @property
    def sin_t(self):
        return math.sin(math.radians(self.tilt_deg))

    @property
    def cos_t(self):
        return math.cos(math.radians(self.tilt_deg))


def load_config(path=None):
    path = path or os.environ.get("REALITY_CONFIG")
    if not path:
        return Config()
    with open(path) as f:
        raw = json.load(f)
    if "markers" in raw:
        raw["markers"] = {int(k): tuple(v) for k, v in raw["markers"].items()}
    return Config(**raw)


@dataclass
class Obj:
    id: str
    kind: str  # ball | guide | obstacle | target
    x: float
    y: float
    angle: float = 0.0  # degrees, long axis from board +x
    length: float = 0.0  # mm, long side (or diameter)
    width: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    conf: float = 1.0
    t: float = 0.0


def angle_diff(a, b):
    """Smallest difference between two undirected line angles in degrees."""
    return abs((a - b + 90) % 180 - 90)


def pose_delta(a, b):
    return math.hypot(a.x - b.x, a.y - b.y), angle_diff(a.angle, b.angle)


@dataclass
class Scene:
    """Debounced world state. `version` bumps only when the layout a plan depends on
    (obstacles, target) changes; the guide is tracked separately because the human moves it."""
    version: int = 0
    ball: Obj | None = None
    guide: Obj | None = None
    target: Obj | None = None
    obstacles: list = field(default_factory=list)
    t: float = 0.0
    move_mm: float = 12.0
    move_deg: float = 6.0
    confirm_frames: int = 3
    _pending: dict = field(default_factory=dict, repr=False)

    def _moved(self, a, b):
        d, da = pose_delta(a, b)
        return d > self.move_mm or (a.kind == "obstacle" and da > self.move_deg)

    def update(self, obs):
        """Fold one confident observation in. Returns a list of human-readable layout changes."""
        self.t = obs.t
        self.ball = obs.ball
        if obs.guide:
            self.guide = obs.guide
        changes = []
        # Target: first sighting sets it, later moves must persist for a few frames.
        if obs.target:
            if self.target is None:
                self.target = obs.target
            elif self._moved(obs.target, self.target):
                if self._confirm("target"):
                    d, _ = pose_delta(obs.target, self.target)
                    changes.append(f"target moved {d:.0f} mm")
                    self.target = obs.target
            else:
                self._pending.pop("target", None)
        # Obstacles: greedy nearest matching against the known set.
        new = obs.obstacles
        if len(new) != len(self.obstacles):
            if not self.obstacles or self._confirm("count"):
                if self.obstacles:
                    changes.append(f"obstacle count {len(self.obstacles)} -> {len(new)}")
                self.obstacles = new
        else:
            self._pending.pop("count", None)
            for o in new:
                old = min(self.obstacles, key=lambda p: math.hypot(p.x - o.x, p.y - o.y))
                o.id = old.id
                if self._moved(o, old):
                    if self._confirm(old.id):
                        d, da = pose_delta(o, old)
                        changes.append(f"{old.id} moved {d:.0f} mm / {da:.0f} deg")
                        self.obstacles[self.obstacles.index(old)] = o
                else:
                    self._pending.pop(old.id, None)
        if changes:
            self.version += 1
        return changes

    def _confirm(self, key):
        self._pending[key] = self._pending.get(key, 0) + 1
        if self._pending[key] >= self.confirm_frames:
            del self._pending[key]
            return True
        return False

    def to_dict(self):
        d = {k: (asdict(v) if isinstance(v, Obj) else v) for k, v in
             (("version", self.version), ("ball", self.ball), ("guide", self.guide), ("target", self.target), ("t", self.t))}
        d["obstacles"] = [asdict(o) for o in self.obstacles]
        return d
