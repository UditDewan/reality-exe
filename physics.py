"""Ball on a tilted board.

`simulate` is a vectorized reduced model (numpy) used for planning and parameter fitting: it rolls
thousands of candidates in lockstep. `newton_rollout` builds the same board, ball, guide and obstacles
as Newton 1.7 rigid-body primitives (one Newton world per candidate) for high-fidelity rehearsal; the
synthetic benchmark uses it as "reality" so the reduced model has a genuine model gap to learn across.
"""
import math

import numpy as np

G = 9810.0  # mm/s^2
PARAM_NAMES = ("mu_roll", "damp", "rest", "mu_wall")
PARAM_LABELS = ("rolling resistance", "speed damping", "bounce", "guide friction")
PRIOR = np.array([0.010, 0.10, 0.50, 0.15])  # belief before any trial: smooth board, lively bounces
BOUNDS = np.array([[0.0, 0.08], [0.0, 1.5], [0.05, 0.9], [0.0, 0.6]])
BALL_K = 0.6  # a/(g sin t) for a thin spherical shell (ping-pong ball) rolling without slipping
WALL_E = 0.4  # board rim restitution; not identified


def boxes_from(objs):
    """Obj list -> (M, 5) array of cx, cy, angle_rad, half_len, half_wid."""
    return np.array([[o.x, o.y, math.radians(o.angle), o.length / 2, o.width / 2] for o in objs]).reshape(-1, 5)


def _box_contact(px, py, vx, vy, cx, cy, ca, sa, hl, hw, r, e, mu):
    """Resolve ball-vs-oriented-box contact. All arguments broadcast over the batch."""
    dx, dy = px - cx, py - cy
    lx, ly = dx * ca + dy * sa, -dx * sa + dy * ca
    ex, ey = lx - np.clip(lx, -hl, hl), ly - np.clip(ly, -hw, hw)
    dist = np.hypot(ex, ey)
    pen = r - dist
    hit = pen > 0
    if not hit.any():
        return px, py, vx, vy, hit
    inside = dist < 1e-9  # centre inside the box (only after a huge step): push out along the shallow axis
    penx, peny = hl - np.abs(lx), hw - np.abs(ly)
    use_x = penx < peny
    safe = np.maximum(dist, 1e-9)
    nlx = np.where(inside, np.where(use_x, np.sign(lx), 0.0), ex / safe)
    nly = np.where(inside, np.where(use_x, 0.0, np.sign(ly)), ey / safe)
    pen = np.where(hit, np.where(inside, r + np.minimum(penx, peny), pen), 0.0)
    nx, ny = nlx * ca - nly * sa, nlx * sa + nly * ca
    px, py = px + nx * pen, py + ny * pen
    vn = vx * nx + vy * ny
    e_eff = np.where(-vn > 30.0, e, 0.0)  # resting/sliding contact does not bounce
    jn = np.where(hit & (vn < 0), -(1 + e_eff) * vn, 0.0)
    tx, ty = -ny, nx
    vt = vx * tx + vy * ty
    dvt = np.minimum(mu * jn, np.abs(vt)) * np.sign(vt)  # Coulomb friction impulse
    return px, py, vx + jn * nx - dvt * tx, vy + jn * ny - dvt * ty, hit


def simulate(cfg, obstacles, guides, params=PRIOR, target=None, start=None, T=4.0, dt=0.004, every=5):
    """Roll the ball once per row of `guides` (N, 3: x, y, angle_deg; NaN row = no guide).

    params: (4,) or (N, 4) in PARAM_NAMES order. Returns path (N, K, 2) sampled every `every` steps,
    dmin (closest approach to target), touch (hit an obstacle), and the sample period.
    """
    guides = np.atleast_2d(np.asarray(guides, float))
    n = len(guides)
    mu_r, damp, e, mu_w = np.broadcast_to(np.asarray(params, float), (n, 4)).T
    sx, sy = start if start is not None else cfg.release
    px, py = np.full(n, float(sx)), np.full(n, float(sy))
    vx, vy = np.zeros(n), np.zeros(n)
    r, w, h = cfg.ball_r, cfg.board_w, cfg.board_h
    g_along, g_n = BALL_K * G * cfg.sin_t, G * cfg.cos_t
    has = ~np.isnan(guides).any(1)
    gx = np.where(has, guides[:, 0], -1e6)
    gy = np.where(has, guides[:, 1], -1e6)
    ga = np.radians(np.nan_to_num(guides[:, 2]))
    gca, gsa = np.cos(ga), np.sin(ga)
    hl, hw = cfg.guide_len / 2, cfg.guide_w / 2
    boxes = boxes_from(obstacles)
    tx, ty = target if target is not None else (np.inf, np.inf)
    done = np.zeros(n, bool)
    touch = np.zeros(n, bool)
    dmin = np.full(n, np.inf)
    path = [np.stack([px, py], 1)]
    for i in range(1, int(T / dt) + 1):
        live = ~done
        speed = np.hypot(vx, vy)
        drop = np.minimum((mu_r * g_n + damp * speed) * dt, speed) / np.maximum(speed, 1e-9)
        vx = (vx - vx * drop) * live
        vy = (vy - vy * drop + g_along * dt) * live
        px, py = px + vx * dt, py + vy * dt
        vx = np.where(((px < r) & (vx < 0)) | ((px > w - r) & (vx > 0)), -WALL_E * vx, vx)
        vy = np.where((py < r) & (vy < 0), -WALL_E * vy, vy)
        px, py = np.clip(px, r, w - r), np.maximum(py, r)
        px, py, vx, vy, _ = _box_contact(px, py, vx, vy, gx, gy, gca, gsa, hl, hw, r, e, mu_w)
        for cx, cy, a, bl, bw in boxes:
            px, py, vx, vy, hit = _box_contact(px, py, vx, vy, cx, cy, math.cos(a), math.sin(a), bl, bw, r, e, mu_w)
            touch |= hit
        if target is not None:
            dmin = np.minimum(dmin, np.where(live, np.hypot(px - tx, py - ty), np.inf))
        done |= py > h - r
        if i % every == 0:
            path.append(np.stack([px, py], 1))
        if done.all():
            break
    path.append(np.stack([px, py], 1))
    return {"path": np.stack(path, 1), "dmin": dmin, "touch": touch, "done": done, "dt": dt * every}


# --- Newton 1.7 rigid-body rehearsal -------------------------------------------------------------

NEWTON_TRUE = {"mu": 0.35, "restitution": 0.25, "mu_rolling": 0.0005}  # hidden "reality" for the benchmark


def newton_rollout(cfg, obstacles, guides, nparams=NEWTON_TRUE, start=None, T=4.0, dt=1 / 500, every=4, device="cpu"):
    """Rehearse each guide pose as its own Newton world. Returns path (N, K, 2) in mm, like `simulate`.

    Board = ground plane under tilted gravity; ball = 2.7 g thin shell on a free body; guide, obstacles
    and the three board rims = static boxes. Units are SI inside Newton.
    """
    import warp as wp
    import newton

    wp.config.quiet = True
    s = 1e-3
    gvec = (0.0, 9.81 * cfg.sin_t, -9.81 * cfg.cos_t)
    r, m = cfg.ball_r * s, 0.0027
    sx, sy = start if start is not None else cfg.release
    mat = newton.ModelBuilder.ShapeConfig(mu=nparams["mu"], restitution=nparams["restitution"],
                                          mu_rolling=nparams["mu_rolling"], density=0.0)
    zq = lambda deg: wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), math.radians(deg))

    def box(b, x, y, deg, length, width, height=30.0):
        b.add_shape_box(-1, xform=wp.transform((x * s, y * s, height / 2 * s), zq(deg)),
                        hx=length / 2 * s, hy=width / 2 * s, hz=height / 2 * s, cfg=mat)

    main = newton.ModelBuilder(gravity=gvec)
    main.add_ground_plane(cfg=mat)
    w, h, t = cfg.board_w, cfg.board_h, 40.0
    box(main, -t / 2, h / 2, 0, t, h + 2 * t)
    box(main, w + t / 2, h / 2, 0, t, h + 2 * t)
    box(main, w / 2, -t / 2, 0, w + 2 * t, t)
    for o in obstacles:
        box(main, o.x, o.y, o.angle, o.length, o.width, 40.0)
    inertia = wp.mat33(np.eye(3) * (2 / 3) * m * r * r)
    for g in np.atleast_2d(guides):
        sub = newton.ModelBuilder(gravity=gvec)
        body = sub.add_body(xform=wp.transform((sx * s, sy * s, r), wp.quat_identity()), mass=m,
                            inertia=inertia, lock_inertia=True)
        sub.add_shape_sphere(body, radius=r, cfg=mat)
        if not np.isnan(g).any():
            box(sub, g[0], g[1], g[2], cfg.guide_len, cfg.guide_w)
        main.add_world(sub)
    model = main.finalize(device=device)
    solver = newton.solvers.SolverXPBD(model, iterations=4, enable_restitution=True)
    s0, s1, ctl = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    path = [s0.body_q.numpy()[:, :2] / s]
    for i in range(1, int(T / dt) + 1):
        s0.clear_forces()
        solver.step(s0, s1, ctl, model.collide(s0), dt)
        s0, s1 = s1, s0
        if i % every == 0:
            p = s0.body_q.numpy()[:, :2] / s
            path.append(p)
            if (p[:, 1] > h - cfg.ball_r).all():
                break
    path = np.stack(path, 1)
    # Freeze each ball where it leaves the board so paths compare like `simulate`.
    for k in range(len(path)):
        out = np.nonzero(path[k, :, 1] > h - cfg.ball_r)[0]
        if len(out):
            path[k, out[0]:] = path[k, out[0]]
    return {"path": path, "dt": dt * every}


def closest_approach(path, point):
    """Min distance from each (dense) path to a point, via segment projection."""
    a, b = path[:, :-1], path[:, 1:]
    ab = b - a
    t = np.clip(((point - a) * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-9), 0, 1)
    return np.linalg.norm(a + ab * t[..., None] - point, axis=-1).min(1)
