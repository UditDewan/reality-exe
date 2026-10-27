"""Synthetic tabletop: renders camera frames (ArUco markers, ball, guide, obstacles, perspective, noise)
so the full OpenCV pipeline runs without hardware. Ball motion comes from Newton with hidden parameters.

    python world.py --print-markers markers.png   # printable markers for the physical board
"""
import math
import sys

import cv2 as cv
import numpy as np

import physics as ph
from scene import Config, Obj
from vision import DICT

S = 1.5  # canvas px per mm
M = 60  # canvas margin, mm
CAM = (960, 720)


class World:
    def __init__(self, cfg=None, seed=0, target=(470, 400), obstacles=((430, 300, 20),), guide=(150, 330, 0),
               engine="newton", truth=None):
        self.cfg = cfg or Config()
        self.rng = np.random.default_rng(seed)
        self.target = target
        self.obstacles = [Obj(f"obstacle-{i + 1}", "obstacle", x, y, a, 60, 45) for i, (x, y, a) in enumerate(obstacles)]
        self.guide = list(guide)
        self.engine = engine
        self.truth = truth or (dict(ph.NEWTON_TRUE) if engine == "newton" else np.array([0.03, 0.4, 0.3, 0.3]))
        self.t, self.occluded = 0.0, False
        self.ball = list(self.cfg.release)
        self.traj, self.t0, self.exited = None, 0.0, False
        cw, ch = (self.cfg.board_w + 2 * M) * S, (self.cfg.board_h + 2 * M) * S
        src = np.float32([[0, 0], [cw, 0], [cw, ch], [0, ch]])
        dst = np.float32([[205, 95], [755, 88], [905, 650], [55, 662]])  # oblique camera looking up the slope
        self.Hcam = cv.getPerspectiveTransform(src, dst)
        self.base = self._board_canvas()

    # --- the "human" -----------------------------------------------------------------------------
    def place_guide(self, x, y, a, noise=True):
        e = self.rng.normal(0, [2.5, 2.5, 1.2]) if noise else 0
        self.guide = list(np.array([x, y, a]) + e)

    def move_obstacle(self, k=0, dx=-70.0, dy=-40.0, da=15.0):
        o = self.obstacles[k]
        o.x, o.y, o.angle = o.x + dx, o.y + dy, o.angle + da

    def release(self):
        """Roll the ball under the hidden physics. Release position jitters like a real gate."""
        start = np.array(self.ball) + self.rng.normal(0, 1.0, 2)
        g = np.array([self.guide])
        if self.engine == "newton":
            r = ph.newton_rollout(self.cfg, self.obstacles, g, self.truth, start=start)
        else:
            r = ph.simulate(self.cfg, self.obstacles, g, self.truth, start=start, dt=0.002, every=4)
        self.traj, self.traj_dt, self.t0, self.exited = r["path"][0], r["dt"], self.t, False
        return self.traj

    def act(self, state, goal):
        """Scripted human that follows the agent's current instruction. Returns True if it did something."""
        if self.traj is not None:
            return False
        if state == "placing" and goal:
            self.place_guide(*goal)
        elif state == "ready":
            self.release()
        elif state == "reset":
            self.reset_ball()
        else:
            return False
        return True

    def reset_ball(self):
        self.traj, self.exited = None, False
        self.ball = list(np.array(self.cfg.release) + self.rng.normal(0, 1.0, 2))

    def tick(self, dt=1 / 30):
        self.t += dt
        if self.traj is not None:
            k = (self.t - self.t0) / self.traj_dt
            if k >= len(self.traj) - 1:
                self.ball = list(self.traj[-1])
                self.exited = self.ball[1] > self.cfg.board_h - self.cfg.ball_r - 1
                self.traj = None
            else:
                i = int(k)
                self.ball = list(self.traj[i] + (self.traj[i + 1] - self.traj[i]) * (k - i))

    # --- rendering ---------------------------------------------------------------------------------
    def _c(self, x, y):
        return (x + M) * S, (y + M) * S

    def _board_canvas(self):
        cfg = self.cfg
        w, h = int((cfg.board_w + 2 * M) * S), int((cfg.board_h + 2 * M) * S)
        img = np.full((h, w, 3), (70, 75, 80), np.uint8)  # table
        x0, y0 = self._c(0, 0)
        x1, y1 = self._c(cfg.board_w, cfg.board_h)
        cv.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), (212, 216, 218), -1)  # white foam board
        noise = self.rng.normal(0, 4, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        cv.rectangle(img, (int(x0), int(y0)), (int(x1), int(y0 - 20 * S)), (60, 60, 60), -1)  # top rim
        side = int(cfg.marker_mm * S)
        for i, (mx, my) in cfg.markers.items():
            self._paste_marker(img, i, mx, my, side)
        return img

    def _paste_marker(self, img, i, mx, my, side):
        m = cv.aruco.generateImageMarker(DICT, i, side)
        cx, cy = self._c(mx, my)
        x, y = int(cx - side / 2), int(cy - side / 2)
        pad = side // 5
        cv.rectangle(img, (x - pad, y - pad), (x + side + pad, y + side + pad), (250, 250, 250), -1)
        img[y:y + side, x:x + side] = cv.cvtColor(m, cv.COLOR_GRAY2BGR)

    def _box(self, img, x, y, a, length, width, color):
        pts = cv.boxPoints((self._c(x, y), (length * S, width * S), a)).astype(np.int32)
        cv.fillPoly(img, [pts], color, cv.LINE_AA)
        cv.polylines(img, [pts], True, tuple(int(c * 0.6) for c in color), 2, cv.LINE_AA)

    def render(self):
        cfg = self.cfg
        img = self.base.copy()
        self._paste_marker(img, cfg.target_id, *self.target, int(cfg.marker_mm * S))
        for o in self.obstacles:
            self._box(img, o.x, o.y, o.angle, o.length, o.width, (180, 40, 200))
        self._box(img, *self.guide, cfg.guide_len, cfg.guide_w, (200, 90, 20))
        if not self.exited:
            cx, cy = self._c(*self.ball)
            r = cfg.ball_r * S
            cv.circle(img, (int(cx + 4), int(cy + 5)), int(r), (150, 150, 150), -1, cv.LINE_AA)  # shadow
            cv.circle(img, (int(cx), int(cy)), int(r), (0, 120, 255), -1, cv.LINE_AA)
            cv.circle(img, (int(cx - r / 3), int(cy - r / 3)), int(r / 4), (170, 210, 255), -1, cv.LINE_AA)
        cam = cv.warpPerspective(img, self.Hcam, CAM, borderValue=(40, 42, 45))
        if self.occluded:  # someone leans over the board: an arm hides the near edge markers
            cv.fillPoly(cam, [np.int32([[0, 720], [0, 470], [520, 500], [960, 470], [960, 720]])], (120, 160, 210), cv.LINE_AA)
            cv.ellipse(cam, (520, 490), (170, 90), 0, 0, 360, (125, 165, 215), -1, cv.LINE_AA)
        cam = cv.GaussianBlur(cam, (3, 3), 0)
        noise = self.rng.normal(0, 3, cam.shape).astype(np.int16)
        return np.clip(cam.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    def truth_scene(self):
        return {"ball": self.ball, "guide": self.guide, "target": self.target,
                "obstacles": [(o.x, o.y, o.angle) for o in self.obstacles], "t": self.t}


def marker_sheet(cfg=None):
    """A4-printable sheet of the five markers at 40 mm (300 dpi)."""
    cfg = cfg or Config()
    px = int(cfg.marker_mm / 25.4 * 300)
    sheet = np.full((3508, 2480), 255, np.uint8)
    for k, i in enumerate(list(cfg.markers) + [cfg.target_id]):
        x, y = 300 + (k % 2) * 1100, 300 + (k // 2) * 1000
        sheet[y:y + px, x:x + px] = cv.aruco.generateImageMarker(DICT, i, px)
        cv.putText(sheet, f"id {i}" + (" TARGET" if i == cfg.target_id else ""), (x, y + px + 80), cv.FONT_HERSHEY_SIMPLEX, 2, 0, 4)
    return sheet


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--print-markers":
        cv.imwrite(sys.argv[2], marker_sheet())
    else:
        cv.imwrite("frame.png", World().render())
