"""Smallest checks that fail if the core logic breaks. Run: python test_core.py  (or: pytest test_core.py)"""
import numpy as np

import identify
import physics as ph
import planner
from controller import Controller
from scene import Config, Obj, Scene
from vision import Vision
from world import World


def test_vision_measures_synthetic_frame():
    w = World()
    o = Vision(w.cfg).observe(w.render(), 0.0)
    assert o.ok and o.markers == [0, 1, 2, 3]
    assert abs(o.ball.x - 300) < 2 and abs(o.ball.y - 45) < 2
    assert abs(o.guide.x - 150) < 3 and abs(o.guide.y - 330) < 3 and abs(o.guide.angle) < 2
    assert abs(o.target.x - 470) < 2 and len(o.obstacles) == 1
    w.occluded = True
    assert not Vision(w.cfg).observe(w.render(), 0.1).ok


def test_physics_guide_deflects_ball():
    r = ph.simulate(Config(), [], np.array([[np.nan] * 3, [300, 200, 30.0]]), ph.PRIOR)
    end = r["path"][:, -1]
    assert abs(end[0, 0] - 300) < 1  # no guide: straight downhill
    assert end[1, 0] > 340  # +30 deg guide sends it right


def test_identify_recovers_hidden_physics():
    cfg, truth, g = Config(), np.array([0.03, 0.4, 0.3, 0.3]), [250, 200, 35.0]
    r = ph.simulate(cfg, [], [g], truth)
    t = np.arange(r["path"].shape[1]) * r["dt"]
    track = np.column_stack([t, r["path"][0]])
    track = track[np.linalg.norm(r["path"][0] - r["path"][0, 0], axis=1) > identify.RELEASE_MM]
    track[:, 1:] += np.random.default_rng(0).normal(0, 1.0, track[:, 1:].shape)
    best, _, rep = identify.fit(cfg, [{"guide": g, "start": cfg.release, "track": track, "obstacles": []}])
    assert rep["fit_err_mm"] < rep["prior_err_mm"]
    g2 = [[330, 230, -30.0]]  # an unseen guide pose: the fitted model must generalise better than the prior
    truth_path = ph.simulate(cfg, [], g2, truth)["path"][0]
    err = lambda p: identify.path_error(truth_path, ph.simulate(cfg, [], g2, p)["path"])[0]
    assert err(best) < err(ph.PRIOR)


def test_planner_finds_valid_hit():
    cfg = Config()
    sc = Scene(target=Obj("target", "target", 470, 400), obstacles=[Obj("o", "obstacle", 430, 300, 20, 60, 45)])
    p = planner.plan(cfg, sc, identify.prior_samples())
    assert p["miss_nominal"] < cfg.target_r
    assert planner.valid(cfg, sc, np.array([p["guide"]]))[0]


def test_moved_obstacle_invalidates_plan_and_occlusion_blocks():
    w = World(engine="reduced")
    c = Controller(w.cfg)
    for _ in range(3):
        w.tick()
        c.step(w.render(), w.t)
    assert c.state == "placing"
    v0 = c.plan["scene_version"]
    w.move_obstacle(0)
    for _ in range(5):
        w.tick()
        c.step(w.render(), w.t)
    assert "plan_invalidated" in [e["event"] for e in c.trace] and c.plan["scene_version"] > v0
    w.occluded = True
    w.tick()
    c.step(w.render(), w.t)
    assert c.blocked and c.trace[-1]["event"] == "clear_view_requested"


def test_blocked_view_mid_roll_is_not_learned_from():
    w = World(engine="reduced", seed=3)
    c = Controller(w.cfg)
    blocked_at = None
    for f in range(400):
        w.tick()
        c.step(w.render(), w.t)
        if c.state == "trial" and blocked_at is None and len(c.track) > 8:
            w.occluded, blocked_at = True, f
        if blocked_at is not None and f - blocked_at > 15:
            w.occluded = False
        if c.trials:
            break
        if f % 10 == 0 and w.traj is None and not w.occluded:
            w.act(c.state, (c.plan or {}).get("guide"))
    assert c.trials, "trial never finished"
    assert "model_update_skipped" in [e["event"] for e in c.trace]
    assert c.trials[0]["hit"] or c.trials[0]["unknown"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
