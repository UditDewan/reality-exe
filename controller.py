"""The agent loop: perceive -> update scene -> plan -> instruct the human -> verify placement from the
camera -> observe the release -> learn physics -> replan. Every decision lands in a trace stamped with
the scene (layout) version it was computed from, so a moved obstacle provably discards stale plans."""
import json
import logging
import math
import time

import numpy as np

import identify
import physics as ph
import planner
from scene import Scene, angle_diff
from vision import Vision, draw

log = logging.getLogger("reality")


class Controller:
    def __init__(self, cfg, adaptive=True, session="local"):
        self.cfg, self.adaptive, self.session = cfg, adaptive, session
        self.vision, self.scene = Vision(cfg), Scene()
        self.params, self.samples = ph.PRIOR.copy(), identify.prior_samples()
        self.state, self.blocked = "calibrating", None
        self.instruction, self.why = "Show the whole board, ball, guide and target to the camera.", ""
        self.plan = self.pred = None
        self.track, self.trials, self.trace, self.data = [], [], [], []
        self.ok_frames, self.last_obs, self.ranges, self.now = 0, None, None, 0.0

    # --- bookkeeping -------------------------------------------------------------------------------
    def event(self, name, **kw):
        e = {"t": round(self.now, 3), "state": self.state, "scene_version": self.scene.version, "event": name, **kw}
        self.trace.append(e)
        log.info(json.dumps({"session": self.session, **e}, default=float))
        return e

    def _start(self):
        b = self.scene.ball
        return (b.x, b.y) if b else self.cfg.release

    # --- main entry ----------------------------------------------------------------------------------
    def step(self, img, t):
        self.now = t
        o = self.vision.observe(img, t)
        self.last_obs = o
        if not o.ok:
            if self.state == "trial":
                self._trial(o, t, [])  # gaps are measured, and later gate the model update
            elif self.blocked != o.reason:
                self.blocked = o.reason
                self.event("clear_view_requested", reason=o.reason)
            return o
        if self.blocked:
            self.event("view_restored", was=self.blocked)
            self.blocked = None
        changes = self.scene.update(o)
        getattr(self, "_" + self.state)(o, t, changes)
        return o

    def _invalidate(self, changes):
        old = self.plan["scene_version"] if self.plan else None
        self.event("plan_invalidated", changes=changes, plan_version=old)
        self._replan(f"{'; '.join(changes)}. Discarded the plan made for layout v{old} before you acted on it.")

    def _replan(self, why):
        sc, cfg = self.scene, self.cfg
        self.plan = planner.plan(cfg, sc, self.samples, current=sc.guide, start=self._start())
        self.plan["scene_version"] = sc.version
        self.pred, self.ok_frames, self.why = None, 0, why
        self.instruction = planner.instruction(sc.guide, self.plan["guide"])
        self.state = "placing"
        self.event("plan", why=why, guide=[round(v, 1) for v in self.plan["guide"]],
                   miss_nominal=round(self.plan["miss_nominal"], 1), miss_worst=round(self.plan["miss_worst"], 1),
                   p_hit=self.plan["p_hit"], candidates=self.plan["candidates"], latency_ms=self.plan["latency_ms"],
                   params=self._params())

    def _params(self):
        return {n: round(float(v), 4) for n, v in zip(ph.PARAM_NAMES, self.params)}

    # --- states --------------------------------------------------------------------------------------
    def _calibrating(self, o, t, changes):
        sc = self.scene
        if sc.ball and sc.guide and sc.target:
            if math.hypot(sc.ball.x - self.cfg.release[0], sc.ball.y - self.cfg.release[1]) > 30:
                self.instruction = "Put the ball in the release gate."
                return
            self.event("scene_ready", obstacles=len(sc.obstacles), target=[round(sc.target.x), round(sc.target.y)])
            self._replan("Initial plan from the measured layout.")

    def _placing(self, o, t, changes):
        if changes:
            return self._invalidate(changes)
        g, goal = self.scene.guide, self.plan["guide"]
        d, da = math.hypot(g.x - goal[0], g.y - goal[1]), angle_diff(g.angle, goal[2])
        self.instruction = planner.instruction(g, goal)
        self.ok_frames = self.ok_frames + 1 if (d < 10 and da < 4 and g.conf > 0.7) else 0
        if self.ok_frames >= 5:
            measured = [g.x, g.y, g.angle]
            self.pred = planner.predict(self.cfg, self.scene, measured, self.params, start=self._start())
            self.pred["guide"] = measured
            self.state = "ready"
            self.instruction = "Placement verified. Release the ball from the gate."
            self.event("placement_verified", error_mm=round(d, 1), error_deg=round(da, 1),
                       predicted_miss=round(self.pred["miss"], 1))

    def _ready(self, o, t, changes):
        if changes:
            return self._invalidate(changes)
        g, m = self.scene.guide, self.pred["guide"]
        if math.hypot(g.x - m[0], g.y - m[1]) > 8 or angle_diff(g.angle, m[2]) > 4:
            self.state, self.ok_frames = "placing", 0
            self.event("guide_disturbed")
            return
        b = self.scene.ball
        if b and math.hypot(b.x - self.pred["path"][0][0], b.y - self.pred["path"][0][1]) > 20:
            self.state, self.track, self.t_seen = "trial", [(t, b.x, b.y)], t
            self.trial_t0 = t
            self.instruction = "Watching the ball..."
            self.event("release_detected")

    def _trial(self, o, t, changes):
        b = o.ball if o.ok else None
        if b or not o.ok:  # a blocked view is not a lost ball; the gap fails the quality gate instead
            self.t_seen = t
        if b:
            self.track.append((t, b.x, b.y))
        lost = t - self.t_seen
        still = len(self.track) > 20 and np.ptp(np.array(self.track[-20:])[:, 1:], 0).max() < 3
        if lost > 0.4 or still or t - self.trial_t0 > 6:
            self._learn()

    def _learn(self):
        cfg, sc = self.cfg, self.scene
        tr = np.array(self.track)
        tgt = np.array([sc.target.x, sc.target.y])
        miss = float(ph.closest_approach(tr[None, :, 1:], tgt)[0]) if len(tr) > 1 else float("inf")
        hit = miss < cfg.target_r
        pred_err = float(identify.path_error(tr[:, 1:], np.array([self.pred["path"]]))[0])
        ok, why = identify.quality(tr, cfg)
        unknown = not hit and "gap" in why  # it may have hit while we could not see it
        before = self._params()
        rec = {"n": len(self.trials) + 1, "hit": hit, "unknown": unknown, "miss_mm": round(miss, 1), "pred_miss_mm": round(self.pred["miss"], 1),
               "traj_err_mm": round(pred_err, 1), "guide": [round(v, 1) for v in self.pred["guide"]],
               "layout_version": sc.version, "params_before": before, "tracked_points": len(tr)}
        if self.adaptive and ok:
            self.data.append({"guide": self.pred["guide"], "start": self.pred["path"][0], "track": tr,
                              "obstacles": list(sc.obstacles)})
            self.params, self.samples, rep = identify.fit(cfg, self.data, self.params)
            self.ranges = rep["ranges"]
            rec.update(params_after=self._params(), fit_err_mm=round(rep["fit_err_mm"], 1))
            self.event("model_updated", before=before, after=self._params(), prior_err_mm=round(rep["prior_err_mm"], 1),
                       fit_err_mm=round(rep["fit_err_mm"], 1), plausible=rep["plausible"])
        else:
            self.event("model_update_skipped", reason="fixed-parameter baseline" if self.adaptive is False else why)
        self.trials.append(rec)
        self.event("trial_result", **{k: rec[k] for k in ("n", "hit", "miss_mm", "pred_miss_mm", "traj_err_mm")})
        self.observed = tr[:, 1:].tolist()
        if hit:
            self.state = "success"
            self.instruction = f"Target reached on attempt {rec['n']}. Move the target or an obstacle for a new challenge."
        elif unknown:
            self.state = "reset"
            self.instruction = "The view was blocked during the roll, so the result is unknown. Put the ball back and release again."
            self.why = f"Trial not used for learning: {why}."
        else:
            self.state = "reset"
            self.instruction = f"Missed by {miss:.0f} mm. Put the ball back in the release gate."
            self.why = f"Observed path was {pred_err:.0f} mm off the prediction; " + (
                "re-fitted physics: " + self._changed(before) + "." if "params_after" in rec else why + ".")

    def _changed(self, before):
        """The two parameters that moved most (relative to their range), in words."""
        old = np.array(list(before.values()))
        rel = np.abs(self.params - old) / (ph.BOUNDS[:, 1] - ph.BOUNDS[:, 0])
        top = [i for i in np.argsort(-rel)[:2] if rel[i] > 0.02] or [int(np.argmax(rel))]
        return ", ".join(f"{ph.PARAM_LABELS[i]} {old[i]:.2f} -> {self.params[i]:.2f}" for i in top)

    def _reset(self, o, t, changes):
        b = self.scene.ball
        if b and math.hypot(b.x - self.cfg.release[0], b.y - self.cfg.release[1]) < 25 and math.hypot(b.vx, b.vy) < 20:
            self.ok_frames += 1
            if self.ok_frames >= 5:
                self._replan(self.why or "Layout changed.")
        else:
            self.ok_frames = 0

    def _success(self, o, t, changes):
        if changes:
            self.event("new_challenge", changes=changes)
            self.state, self.ok_frames, self.why = "reset", 0, f"New challenge: {'; '.join(changes)}."
            self.instruction = "New layout seen. Put the ball back in the release gate."

    # --- presentation ----------------------------------------------------------------------------------
    def overlay(self):
        p = self.plan or {}
        ov = {"alternatives": p.get("alternatives", [])[:3], "ghosts": p.get("ghosts", []),
              "pred_path": (self.pred or p).get("path", p.get("pred_path"))}
        if self.state == "placing":
            ov["goal"] = p.get("guide")
        if self.state in ("trial",):
            ov["observed"] = [q[1:] for q in self.track]
        elif self.state in ("reset", "success"):
            ov["observed"] = getattr(self, "observed", [])
        return ov

    def annotate(self, img):
        return draw(img, self.last_obs, self.cfg, self.overlay()) if self.last_obs else img

    def status(self):
        p = self.plan or {}
        return {
            "state": self.state, "instruction": self.instruction, "why": self.why, "blocked": self.blocked,
            "scene": self.scene.to_dict(), "obs": self.last_obs.summary() if self.last_obs else None,
            "plan": {k: p.get(k) for k in ("guide", "miss_nominal", "miss_worst", "p_hit", "latency_ms", "candidates", "scene_version")},
            "overlay": self.overlay(), "params": self._params(), "ranges": self.ranges,
            "trials": self.trials, "events": self.trace[-15:], "board": [self.cfg.board_w, self.cfg.board_h, self.cfg.tilt_deg],
        }
