"""Search guide poses. Score = target miss across plausible physics + obstacle contact + human effort."""
import itertools
import math
import time

import numpy as np

import physics as ph
from scene import angle_diff


def _seg_points(g, cfg, n=9):
    """(N, 3) guide poses -> (N, n, 2) points along each guide centreline."""
    a = np.radians(g[:, 2])
    s = np.linspace(-cfg.guide_len / 2, cfg.guide_len / 2, n)
    return np.stack([g[:, :1] + np.cos(a)[:, None] * s, g[:, 1:2] + np.sin(a)[:, None] * s], -1)


def _dist_to_box(p, o):
    a = math.radians(o.angle)
    d = p - (o.x, o.y)
    lx, ly = d[..., 0] * math.cos(a) + d[..., 1] * math.sin(a), -d[..., 0] * math.sin(a) + d[..., 1] * math.cos(a)
    return np.hypot(lx - np.clip(lx, -o.length / 2, o.length / 2), ly - np.clip(ly, -o.width / 2, o.width / 2))


def valid(cfg, scene, g):
    pts = _seg_points(g, cfg)
    hw = cfg.guide_w / 2
    ok = (pts[..., 0].min(1) > hw) & (pts[..., 0].max(1) < cfg.board_w - hw) & (pts[..., 1].min(1) > hw) & (pts[..., 1].max(1) < cfg.board_h - hw)
    for o in scene.obstacles:
        ok &= _dist_to_box(pts, o).min(1) > hw + 8
    ok &= np.linalg.norm(pts - cfg.release, axis=-1).min(1) > cfg.ball_r + hw + 15
    if scene.target:
        ok &= np.linalg.norm(pts - (scene.target.x, scene.target.y), axis=-1).min(1) > cfg.target_r + hw
    return ok


def grid(cfg, step=30.0, ang=7.5):
    x0, y0, x1, y1 = cfg.guide_region
    angles = [a for a in np.arange(-60, 60.1, ang) if abs(a) >= 10]
    return np.array(list(itertools.product(np.arange(x0, x1 + 1, step), np.arange(y0, y1 + 1, step), angles)), float)


def plan(cfg, scene, samples, current=None, start=None):
    """samples: (S, 4) plausible parameter sets, first row = best estimate."""
    t0 = time.perf_counter()
    tgt = (scene.target.x, scene.target.y)
    sim = lambda g, p: ph.simulate(cfg, scene.obstacles, g, p, target=tgt, start=start)
    g = grid(cfg)
    g = g[valid(cfg, scene, g)]
    r = sim(g, samples[0])
    coarse = r["dmin"] + 15 * r["touch"]
    top = g[np.argsort(coarse)[:24]]
    local = (top[:6, None, :] + np.array(list(itertools.product((-15, 0, 15), (-15, 0, 15), (-4, 0, 4))))[None]).reshape(-1, 3)
    pool = np.unique(np.vstack([top, local[valid(cfg, scene, local)]]), axis=0)
    s = len(samples)
    rr = sim(np.repeat(pool, s, 0), np.tile(samples, (len(pool), 1)))
    d = rr["dmin"].reshape(-1, s)
    touch = rr["touch"].reshape(-1, s).any(1)
    robust = 0.5 * d.mean(1) + 0.5 * d.max(1) + 15 * touch
    effort = 0.0
    if current is not None:
        effort = 0.03 * np.hypot(pool[:, 0] - current.x, pool[:, 1] - current.y) + 0.1 * angle_diff(pool[:, 2], current.angle)
    total = robust + effort
    order = np.argsort(total)
    b = order[0]
    paths = rr["path"].reshape(len(pool), s, -1, 2)
    return {
        "guide": pool[b].tolist(),
        "pred_path": paths[b, 0].tolist(),
        "ghosts": paths[b, 1:].tolist(),  # same plan under the other plausible physics
        "alternatives": [paths[k, 0].tolist() for k in order[1:5]],
        "alt_guides": pool[order[1:5]].tolist(),
        "miss_nominal": float(d[b, 0]),
        "miss_worst": float(d[b].max()),
        "p_hit": float((d[b] < cfg.target_r).mean()),
        "touches_obstacle": bool(touch[b]),
        "candidates": int(len(g) + len(pool) * s),
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        "dt": rr["dt"],
    }


def predict(cfg, scene, guide, params, start=None):
    """Prediction for the guide as actually placed (measured), not as instructed."""
    r = ph.simulate(cfg, scene.obstacles, [guide], params, target=(scene.target.x, scene.target.y), start=start)
    return {"path": r["path"][0].tolist(), "miss": float(r["dmin"][0]), "dt": r["dt"]}


def instruction(cur, goal):
    """Human-readable move from the measured guide pose to the planned pose."""
    if cur is None:
        return f"Place the blue guide centred at ({goal[0]:.0f}, {goal[1]:.0f}) mm, angled {goal[2]:+.0f} deg."
    dx, dy = goal[0] - cur.x, goal[1] - cur.y
    da = (goal[2] - cur.angle + 90) % 180 - 90
    parts = []
    if abs(dx) >= 5:
        parts.append(f"{abs(dx):.0f} mm {'right' if dx > 0 else 'left'}")
    if abs(dy) >= 5:
        parts.append(f"{abs(dy):.0f} mm {'downhill' if dy > 0 else 'uphill'}")
    move = "Slide the blue guide " + " and ".join(parts) if parts else "Keep the guide where it is"
    if abs(da) >= 2:
        move += f", then rotate it {abs(da):.0f} deg {'clockwise' if da > 0 else 'counter-clockwise'}"
    return move + "."
