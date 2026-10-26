"""OpenCV 5 perception: ArUco board registration -> rectified top-down board -> HSV segmentation
-> ball / guide / obstacle / target poses in millimetres, plus Kalman-filtered ball velocity."""
import math
import time
from dataclasses import dataclass, field

import cv2 as cv
import numpy as np

from scene import Obj

DICT = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)


@dataclass
class Observation:
    t: float
    ok: bool = False
    reason: str = ""
    markers: list = field(default_factory=list)  # visible board-marker ids
    reproj_mm: float = float("nan")
    sharpness: float = 0.0
    ball: Obj | None = None
    guide: Obj | None = None
    target: Obj | None = None
    obstacles: list = field(default_factory=list)
    H: np.ndarray | None = None  # image px -> board mm
    corners: tuple = ()
    ids: list = field(default_factory=list)
    ms: float = 0.0

    def summary(self):
        f = lambda o: None if o is None else {k: round(v, 1) if isinstance(v, float) else v for k, v in vars(o).items()}
        return {"t": round(self.t, 3), "ok": self.ok, "reason": self.reason, "markers": self.markers,
                "reproj_mm": None if math.isnan(self.reproj_mm) else round(self.reproj_mm, 2),
                "sharpness": round(self.sharpness, 1), "ms": round(self.ms, 1), "ball": f(self.ball),
                "guide": f(self.guide), "target": f(self.target), "obstacles": [f(o) for o in self.obstacles]}


class Vision:
    def __init__(self, cfg, px_per_mm=1.0):
        self.cfg, self.s = cfg, px_per_mm
        p = cv.aruco.DetectorParameters()
        p.cornerRefinementMethod = cv.aruco.CORNER_REFINE_SUBPIX
        self.det = cv.aruco.ArucoDetector(DICT, p)
        h = cfg.marker_mm / 2
        self.board_pts = {i: np.float32([[x - h, y - h], [x + h, y - h], [x + h, y + h], [x - h, y + h]])
                          for i, (x, y) in cfg.markers.items()}
        self.kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        self.kf, self.kf_t = None, None

    def observe(self, img, t):
        t0 = time.perf_counter()
        cfg = self.cfg
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        corners, ids, _ = self.det.detectMarkers(gray)
        ids = [] if ids is None else ids.ravel().tolist()
        o = Observation(t=t, corners=corners, ids=ids, sharpness=float(cv.Laplacian(gray, cv.CV_64F).var()))
        o.markers = sorted(i for i in ids if i in self.board_pts)
        if len(o.markers) < cfg.min_markers:
            o.reason = f"only {len(o.markers)} of 4 board markers visible"
            return self._done(o, t0)
        src = np.concatenate([corners[ids.index(i)].reshape(4, 2) for i in o.markers])
        dst = np.concatenate([self.board_pts[i] for i in o.markers])
        H, _ = cv.findHomography(src, dst, cv.RANSAC, 4.0)
        if H is None:
            o.reason = "board homography failed"
            return self._done(o, t0)
        o.reproj_mm = float(np.linalg.norm(cv.perspectiveTransform(src[None], H)[0] - dst, axis=1).mean())
        if o.reproj_mm > 5.0:
            o.reason = f"board registration error {o.reproj_mm:.1f} mm"
            return self._done(o, t0)
        o.H = H
        if cfg.target_id in ids:
            c = cv.perspectiveTransform(corners[ids.index(cfg.target_id)].reshape(1, 4, 2), H)[0]
            o.target = Obj("target", "target", float(c[:, 0].mean()), float(c[:, 1].mean()), length=cfg.marker_mm, t=t)
        s = self.s
        top = cv.warpPerspective(img, np.diag([s, s, 1.0]) @ H, (int(cfg.board_w * s), int(cfg.board_h * s)))
        hsv = cv.cvtColor(top, cv.COLOR_BGR2HSV)
        o.ball = self._ball(self._mask(hsv, "ball"), t)
        boxes = self._boxes(self._mask(hsv, "guide"), 0.25 * cfg.guide_len * cfg.guide_w, "guide", t)
        if boxes:
            o.guide = max(boxes, key=lambda b: b.length * b.width)
            o.guide.conf = min(1.0, o.guide.length * o.guide.width / (cfg.guide_len * cfg.guide_w))
        o.obstacles = self._boxes(self._mask(hsv, "obstacle"), 600.0, "obstacle", t)
        o.ok = True
        return self._done(o, t0)

    def _done(self, o, t0):
        o.ms = (time.perf_counter() - t0) * 1000
        return o

    def _mask(self, hsv, name):
        lo, hi = self.cfg.hsv[name]
        m = cv.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        return cv.morphologyEx(cv.morphologyEx(m, cv.MORPH_OPEN, self.kernel), cv.MORPH_CLOSE, self.kernel)

    def _ball(self, mask, t):
        s, r = self.s, self.cfg.ball_r
        expect = math.pi * (r * s) ** 2
        best = None
        for c in cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]:
            area = cv.contourArea(c)
            if not 0.3 * expect < area < 2.5 * expect:
                continue
            _, rad = cv.minEnclosingCircle(c)
            circ = area / (math.pi * rad * rad)
            if circ < 0.45:  # motion blur stretches the ball; below this it is not a ball
                continue
            score = circ - 0.3 * abs(math.log(area / expect))
            if best is None or score > best[0]:
                m = cv.moments(c)
                best = (score, m["m10"] / m["m00"] / s, m["m01"] / m["m00"] / s, circ)
        if best is None:
            return None
        _, x, y, circ = best
        vx, vy = self._track(x, y, t)
        return Obj("ball", "ball", x, y, length=2 * r, width=2 * r, vx=vx, vy=vy, conf=min(1.0, circ / 0.85), t=t)

    def _track(self, x, y, t):
        """Constant-velocity cv.KalmanFilter in board mm; re-initialises on gaps or jumps."""
        z = np.array([[x], [y]], np.float32)
        if self.kf is not None and 0 < t - self.kf_t < 0.5:
            dt = t - self.kf_t
            self.kf.transitionMatrix = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
            pred = self.kf.predict()
            if math.hypot(pred[0, 0] - x, pred[1, 0] - y) < 80:
                st = self.kf.correct(z)
                self.kf_t = t
                return float(st[2, 0]), float(st[3, 0])
        kf = cv.KalmanFilter(4, 2)
        kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        kf.processNoiseCov = np.diag([0.5, 0.5, 300.0, 300.0]).astype(np.float32)  # ponytail: tuned for 30 fps
        kf.measurementNoiseCov = np.diag([2.0, 2.0]).astype(np.float32)
        kf.errorCovPost = np.diag([4.0, 4.0, 1e4, 1e4]).astype(np.float32)
        kf.statePost = np.array([[x], [y], [0], [0]], np.float32)
        self.kf, self.kf_t = kf, t
        return 0.0, 0.0

    def _boxes(self, mask, min_area, kind, t):
        out = []
        for c in cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]:
            area = cv.contourArea(c) / self.s ** 2
            if area < min_area:
                continue
            (cx, cy), (w, h), a = cv.minAreaRect(c)
            if w < h:
                w, h, a = h, w, a + 90
            a = (a + 90) % 180 - 90
            s = self.s
            out.append(Obj(f"{kind}-{len(out) + 1}", kind, cx / s, cy / s, a, w / s, h / s,
                           conf=min(1.0, area / max(w * h / s ** 2, 1)), t=t))
        return out


# --- overlay ------------------------------------------------------------------------------------

def _proj(Hinv, pts):
    pts = np.asarray(pts, np.float32).reshape(-1, 1, 2)
    return cv.perspectiveTransform(pts, Hinv).reshape(-1, 2).astype(np.int32)


def _box_pts(x, y, a, length, width):
    return cv.boxPoints(((x, y), (length, width), a))


def draw(img, o, cfg, overlay=None):
    """Annotate the camera frame: registration, detections, ghost trajectories, the instructed pose."""
    out = img.copy()
    if len(o.ids):
        cv.aruco.drawDetectedMarkers(out, o.corners, np.array(o.ids).reshape(-1, 1))
    if o.H is None:
        return out
    Hinv = np.linalg.inv(o.H)
    ghost = out.copy()
    ov = overlay or {}
    for p in ov.get("alternatives", []):
        cv.polylines(ghost, [_proj(Hinv, p)], False, (200, 200, 90), 2, cv.LINE_AA)
    for p in ov.get("ghosts", []):
        cv.polylines(ghost, [_proj(Hinv, p)], False, (255, 220, 120), 2, cv.LINE_AA)
    if ov.get("goal"):
        cv.fillPoly(ghost, [_proj(Hinv, _box_pts(*ov["goal"], cfg.guide_len, cfg.guide_w))], (255, 160, 40))
    cv.addWeighted(ghost, 0.45, out, 0.55, 0, out)
    cv.polylines(out, [_proj(Hinv, [(0, 0), (cfg.board_w, 0), (cfg.board_w, cfg.board_h), (0, cfg.board_h)])], True, (80, 220, 80), 2)
    if ov.get("pred_path"):
        cv.polylines(out, [_proj(Hinv, ov["pred_path"])], False, (255, 255, 0), 3, cv.LINE_AA)
    if ov.get("observed"):
        for q in _proj(Hinv, ov["observed"]):
            cv.circle(out, tuple(int(v) for v in q), 3, (0, 0, 255), -1)
    if o.target:
        c = _proj(Hinv, [(o.target.x, o.target.y)])[0]
        cv.circle(out, tuple(int(v) for v in c), 18, (0, 255, 0), 3)
    if o.guide:
        cv.polylines(out, [_proj(Hinv, _box_pts(o.guide.x, o.guide.y, o.guide.angle, o.guide.length, o.guide.width))], True, (255, 80, 0), 2)
    for ob in o.obstacles:
        cv.polylines(out, [_proj(Hinv, _box_pts(ob.x, ob.y, ob.angle, ob.length, ob.width))], True, (255, 0, 255), 2)
    if o.ball:
        c = _proj(Hinv, [(o.ball.x, o.ball.y)])[0]
        cv.circle(out, tuple(int(v) for v in c), 14, (0, 140, 255), 2)
    return out
