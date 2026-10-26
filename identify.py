"""Fit effective reduced-model parameters to every observed trial so far, only from good tracking.

Each trial constrains the physics twice: by the shape of the path and by its timing (the Kalman-
tracked timestamps pin down speed, which path shape alone cannot)."""
import itertools

import numpy as np

import physics as ph

RELEASE_MM = 20.0  # the controller declares a release at this displacement; timing is aligned there


def path_error(obs, paths):
    """Mean distance (mm) from each observed point (K, 2) to each simulated path (N, M, 2)."""
    d2 = (obs ** 2).sum(-1)[None, :, None] + (paths ** 2).sum(-1)[:, None, :] - 2 * np.einsum("kd,nmd->nkm", obs, paths)
    return np.sqrt(np.maximum(d2, 0).min(2)).mean(1)


def timing_error(track, paths, dt):
    """Mean distance between observed and simulated positions at matching times since release."""
    n, m = paths.shape[:2]
    moved = np.linalg.norm(paths - paths[:, :1], axis=-1) > RELEASE_MM
    i0 = np.where(moved.any(1), moved.argmax(1), m - 1)
    f = np.clip(i0[:, None] + (track[None, :, 0] - track[0, 0]) / dt, 0, m - 1)  # (N, K) fractional index
    lo = np.floor(f).astype(int)
    hi = np.minimum(lo + 1, m - 1)
    w = (f - lo)[..., None]
    rows = np.arange(n)[:, None]
    sim = paths[rows, lo] * (1 - w) + paths[rows, hi] * w
    return np.linalg.norm(sim - track[None, :, 1:], axis=-1).mean(1)


def quality(track, cfg):
    """track: (K, 3) rows of t, x, y. Returns (ok, reason)."""
    if len(track) < 10:
        return False, f"only {len(track)} ball fixes"
    span = np.ptp(track[:, 1:], axis=0).max()
    if span < 30:  # a ball that sticks after 50 mm is strong evidence about friction; keep it
        return False, f"track spans {span:.0f} mm"
    gap = np.diff(track[:, 0]).max()
    if gap > 0.25:
        return False, f"tracking gap of {gap * 1000:.0f} ms"
    return True, "ok"


def error(cfg, trials, P):
    """Mean over trials of path + timing error for each parameter row of P."""
    total = np.zeros(len(P))
    for tr in trials:
        r = ph.simulate(cfg, tr["obstacles"], np.tile(tr["guide"], (len(P), 1)), P, start=tr["start"])
        k = tr["track"][:: max(1, len(tr["track"]) // 40)]
        total += 0.5 * path_error(k[:, 1:], r["path"]) + 0.5 * timing_error(k, r["path"], r["dt"])
    return total / len(trials)


def fit(cfg, trials, prior=ph.PRIOR):
    """Grid-search the 4 effective parameters, refine, and return (best, plausible samples, report)."""
    lo, hi = ph.BOUNDS.T
    counts = np.array([7, 5, 4, 7])
    grid = np.array(list(itertools.product(*[np.linspace(lo[i], hi[i], c) for i, c in enumerate(counts)])))
    err = error(cfg, trials, grid)
    step = (hi - lo) / (counts - 1)
    fine = np.clip(grid[err.argmin()] + np.array(list(itertools.product((-0.5, 0, 0.5), repeat=4))) * step, lo, hi)
    grid, err = np.vstack([grid, fine]), np.concatenate([err, error(cfg, trials, fine)])
    best, e0 = grid[err.argmin()], err.min()
    plaus = grid[err <= e0 * 1.25 + 1.0]  # ponytail: tolerance heuristic, swap for a likelihood once noise is characterised
    picks = np.array([best] + [plaus[plaus[:, i].argmin()] for i in range(4)] + [plaus[plaus[:, i].argmax()] for i in range(4)])
    samples = np.vstack([best, np.unique(picks[1:][~np.all(np.isclose(picks[1:], best), 1)], axis=0)])
    return best, samples, {
        "fit_err_mm": float(e0), "prior_err_mm": float(error(cfg, trials, np.atleast_2d(prior))[0]),
        "plausible": int(len(plaus)), "evaluated": int(len(grid)), "trials": len(trials),
        "ranges": {n: [float(plaus[:, i].min()), float(plaus[:, i].max())] for i, n in enumerate(ph.PARAM_NAMES)},
    }


def prior_samples(prior=ph.PRIOR, spread=0.2):
    """Before any trial the agent is (over)confident: +-20% around the prior."""
    lo, hi = ph.BOUNDS.T
    return np.clip(np.vstack([prior, prior * (1 - spread), prior * (1 + spread)]), lo, hi)
