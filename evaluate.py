"""Reproducible benchmark on the synthetic tabletop (Newton plays reality, the agent only sees rendered
camera frames). Writes eval/results.json, eval/results.md, eval/trace_example.json, eval/first_milestone.mp4.

    python evaluate.py            # full run (~10 min on a laptop CPU)
    python evaluate.py --quick    # 3 layouts
"""
import json
import math
import os
import sys
import time

import cv2 as cv
import numpy as np

import physics as ph
from controller import Controller
from scene import angle_diff
from world import World

OUT = "eval"


def layout(seed):
    rng = np.random.default_rng(seed)
    target = (float(rng.uniform(90, 510)), float(rng.uniform(370, 415)))
    obstacles = []
    while len(obstacles) < rng.integers(1, 3):
        o = (float(rng.uniform(90, 510)), float(rng.uniform(190, 340)), float(rng.uniform(-40, 40)))
        if math.hypot(o[0] - target[0], o[1] - target[1]) > 110 and all(math.hypot(o[0] - p[0], o[1] - p[1]) > 120 for p in obstacles):
            obstacles.append(o)
    truth = {k: v * float(rng.uniform(0.75, 1.25)) for k, v in ph.NEWTON_TRUE.items()}  # a different "surface" each time
    return dict(target=target, obstacles=tuple(obstacles), guide=(120.0, 150.0, 0.0), truth=truth)


def hud(img, c):
    out = c.annotate(img)
    cv.rectangle(out, (0, 0), (out.shape[1], 64), (20, 20, 20), -1)
    cv.putText(out, f"REALITY.EXE  [{c.state}]  layout v{c.scene.version}  trials {len(c.trials)}", (12, 24), cv.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2)
    cv.putText(out, (c.blocked and f"CLEAR VIEW: {c.blocked}") or c.instruction[:96], (12, 52), cv.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return out


def episode(world, ctrl, max_trials=5, max_frames=4000, script=None, video=None, errs=None):
    """Drive world + agent until success or max_trials. `script(frame, world, ctrl)` injects disturbances."""
    cool = 0
    for f in range(max_frames):
        world.tick()
        img = world.render()
        o = ctrl.step(img, world.t)
        if errs is not None and o.ok:
            if o.ball and not world.exited:
                errs["ball"].append(math.hypot(o.ball.x - world.ball[0], o.ball.y - world.ball[1]))
            if o.guide:
                errs["guide_mm"].append(math.hypot(o.guide.x - world.guide[0], o.guide.y - world.guide[1]))
                errs["guide_deg"].append(angle_diff(o.guide.angle, world.guide[2]))
            errs["ms"].append(o.ms)
        if video is not None:
            video.write(hud(img, ctrl))
        if script:
            script(f, world, ctrl)
        cool -= 1
        if cool <= 0 and not world.occluded and world.act(ctrl.state, (ctrl.plan or {}).get("guide")):
            cool = 12
        if ctrl.state == "success" or len(ctrl.trials) >= max_trials:
            break
    return ctrl


def first_hit(c):
    return next((t["n"] for t in c.trials if t["hit"]), None)


def stale_actions(trace):
    """Human actions (placement verified / release) taken against a plan from an older layout version."""
    v, n = None, 0
    for e in trace:
        if e["event"] == "plan":
            v = e["scene_version"]
        elif e["event"] in ("placement_verified", "release_detected") and e["scene_version"] != v:
            n += 1
    return n


def main(quick=False):
    os.makedirs(OUT, exist_ok=True)
    for f in ("first_milestone.mp4", "_clip.mp4"):
        if os.path.exists(f"{OUT}/{f}"):
            os.remove(f"{OUT}/{f}")
    seeds = range(3) if quick else range(10)
    res = {"adaptive": [], "fixed": [], "obstacle_move": [], "occlusion": {}, "latency_ms": [], "vision": {}}
    errs = {"ball": [], "guide_mm": [], "guide_deg": [], "ms": []}
    t0 = time.time()
    for s in seeds:
        L = layout(s)
        for mode in ("adaptive", "fixed"):
            w = World(seed=100 + s, target=L["target"], obstacles=L["obstacles"], guide=L["guide"], truth=L["truth"])
            vid = None
            if mode == "adaptive" and not os.path.exists(f"{OUT}/first_milestone.mp4"):
                vid = cv.VideoWriter(f"{OUT}/_clip.mp4", cv.VideoWriter_fourcc(*"mp4v"), 30, (960, 720))
            c = episode(w, Controller(w.cfg, adaptive=mode == "adaptive", session=f"{mode}-{s}"), video=vid, errs=errs)
            if vid is not None:  # keep the first episode that shows prediction -> real miss -> corrected hit
                vid.release()
                if (first_hit(c) or 0) >= 2:
                    os.replace(f"{OUT}/_clip.mp4", f"{OUT}/first_milestone.mp4")
                    json.dump(c.trace, open(f"{OUT}/trace_example.json", "w"), indent=1, default=float)
            res[mode].append({"seed": s, "first_hit": first_hit(c), "trials": c.trials})
            res["latency_ms"] += [e["latency_ms"] for e in c.trace if e["event"] == "plan"]
            print(f"[{time.time() - t0:5.0f}s] seed {s} {mode:8s} first hit: {first_hit(c)}  misses: {[t['miss_mm'] for t in c.trials]}", flush=True)

        # Recovery: the obstacle moves while the human is being told where to put the guide.
        def bump(f, w, c, st={}):
            if c.state == "placing" and not st and len(c.trials) == 0:
                st["moved"] = True
                w.move_obstacle(0)
        w = World(seed=200 + s, target=L["target"], obstacles=L["obstacles"], guide=L["guide"], truth=L["truth"])
        c = episode(w, Controller(w.cfg, session=f"move-{s}"), script=bump)
        inv = [e for e in c.trace if e["event"] == "plan_invalidated"]
        plans = [e for e in c.trace if e["event"] == "plan"]
        res["obstacle_move"].append({"seed": s, "detected": bool(inv), "replanned": len(plans) >= 2 and plans[1]["scene_version"] > plans[0]["scene_version"],
                                     "stale_actions": stale_actions(c.trace), "first_hit": first_hit(c)})
        print(f"        obstacle moved: detected={bool(inv)} first hit: {first_hit(c)}", flush=True)

    # Occlusion: an arm covers the near markers right after placement is verified, and mid-trial.
    def occlude(f, w, c, st={"n": 0}):
        if c.state == "ready" and st["n"] == 0:
            w.occluded, st["n"], st["f"] = True, 1, f
        elif st["n"] == 1 and f - st["f"] > 30:
            w.occluded, st["n"] = False, 2
        elif c.state == "trial" and st["n"] == 2 and len(c.track) > 12:
            w.occluded, st["n"], st["f"] = True, 3, f
        elif st["n"] == 3 and f - st["f"] > 12:
            w.occluded, st["n"] = False, 4
    L = layout(0)
    w = World(seed=300, target=L["target"], obstacles=L["obstacles"], guide=L["guide"], truth=L["truth"])
    c = episode(w, Controller(w.cfg, session="occlusion"), script=occlude, max_trials=2)
    ev = [e["event"] for e in c.trace]
    res["occlusion"] = {
        "clear_view_requested": "clear_view_requested" in ev, "view_restored": "view_restored" in ev,
        "released_while_blocked": False,  # the scripted human never acts while occluded; the agent issues no new instruction
        "mid_trial_update_skipped": any(e["event"] == "model_update_skipped" for e in c.trace),
        "occluded_roll_outcome": next(("hit (observed)" if t["hit"] else "unknown" if t.get("unknown") else "miss"
                                       for t in c.trials if t["tracked_points"] and not t.get("params_after")), "n/a"),
        "events": [e for e in c.trace if e["event"] in ("clear_view_requested", "view_restored", "model_update_skipped")],
    }
    # First technical checkpoint: do repeated releases of one setup land predictably? (gate jitter 1 mm)
    w = World(seed=400)
    w.place_guide(205, 320, 26, noise=False)
    ends, close = [], []
    for _ in range(3 if quick else 8):
        w.reset_ball()
        p = w.release()
        ends.append(p[-1])
        close.append(float(ph.closest_approach(p[None], np.array(w.target))[0]))
    ends = np.array(ends)
    res["repeatability"] = {"releases": len(ends), "exit_x_std_mm": float(ends[:, 0].std()), "exit_y_std_mm": float(ends[:, 1].std()),
                            "target_dist_mean_mm": float(np.mean(close)), "target_dist_std_mm": float(np.std(close))}
    if os.path.exists(f"{OUT}/_clip.mp4"):
        os.remove(f"{OUT}/_clip.mp4")
    res["vision"] = {k: {"mean": float(np.mean(v)), "p95": float(np.percentile(v, 95)), "n": len(v)} for k, v in errs.items()}
    json.dump(res, open(f"{OUT}/results.json", "w"), indent=1, default=float)
    write_md(res)
    print(open(f"{OUT}/results.md", encoding="utf-8").read())


def write_md(r):
    def rate(runs, k):
        return sum(1 for x in runs if x["first_hit"] and x["first_hit"] <= k) / len(runs)

    def traj_err(runs, i):
        v = [x["trials"][i]["traj_err_mm"] for x in runs if len(x["trials"]) > i]
        return f"{np.mean(v):.1f}" if v else "-"

    a, f = r["adaptive"], r["fixed"]
    lat = np.array(r["latency_ms"])
    lines = [
        "# Evaluation results (synthetic tabletop, Newton 1.7 as reality)", "",
        f"Layouts: {len(a)}. Each layout has its own hidden surface physics. The agent sees only rendered camera frames.", "",
        "| Metric | Adaptive (fits physics) | Fixed prior physics |", "|---|---|---|",
        *[f"| Target hit within {k} trial{'s' if k > 1 else ''} | {rate(a, k):.0%} | {rate(f, k):.0%} |" for k in (1, 2, 3, 5)],
        *[f"| Trajectory error, trial {i + 1} (mm) | {traj_err(a, i)} | {traj_err(f, i)} |" for i in range(3)],
        "", f"Planning latency: median {np.median(lat):.0f} ms, p95 {np.percentile(lat, 95):.0f} ms over {len(lat)} plans.", "",
        "## Recovery after an obstacle moves mid-instruction", "",
        f"Detected and invalidated: {sum(x['detected'] for x in r['obstacle_move'])}/{len(r['obstacle_move'])}; "
        f"replanned on the new layout version: {sum(bool(x['replanned']) for x in r['obstacle_move'])}/{len(r['obstacle_move'])}; "
        f"human actions taken on a stale plan: {sum(x['stale_actions'] for x in r['obstacle_move'])}; "
        f"target later hit: {sum(1 for x in r['obstacle_move'] if x['first_hit'])}/{len(r['obstacle_move'])}.", "",
        "## Occluded camera", "",
        f"Clear view requested: {r['occlusion']['clear_view_requested']}; resumed after view restored: {r['occlusion']['view_restored']}; "
        f"model update skipped for the trial with a tracking gap: {r['occlusion']['mid_trial_update_skipped']}; "
        f"outcome of the occluded roll: {r['occlusion']['occluded_roll_outcome']} "
        "(an unobserved miss is reported as unknown, see test_core.py).", "",
        "## Repeatability (first technical checkpoint)", "",
        f"{r['repeatability']['releases']} Newton releases of one fixed setup with 1 mm gate jitter: exit point std "
        f"{r['repeatability']['exit_x_std_mm']:.1f} mm (x) / {r['repeatability']['exit_y_std_mm']:.1f} mm (y); closest approach to target "
        f"{r['repeatability']['target_dist_mean_mm']:.1f} ± {r['repeatability']['target_dist_std_mm']:.1f} mm.", "",
        "## OpenCV measurement accuracy (vs. renderer ground truth)", "",
        "| Quantity | Mean | p95 | Samples |", "|---|---|---|---|",
        *[f"| {k} | {v['mean']:.2f} | {v['p95']:.2f} | {v['n']} |" for k, v in r["vision"].items()],
    ]
    open(f"{OUT}/results.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
