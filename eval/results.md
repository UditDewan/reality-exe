# Evaluation results (synthetic tabletop, Newton 1.7 as reality)

Layouts: 10. Each layout has its own hidden surface physics. The agent sees only rendered camera frames.

| Metric | Adaptive (fits physics) | Fixed prior physics |
|---|---|---|
| Target hit within 1 trial | 40% | 40% |
| Target hit within 2 trials | 90% | 40% |
| Target hit within 3 trials | 100% | 40% |
| Target hit within 5 trials | 100% | 40% |
| Trajectory error, trial 1 (mm) | 12.0 | 12.0 |
| Trajectory error, trial 2 (mm) | 3.2 | 9.8 |
| Trajectory error, trial 3 (mm) | 1.6 | 10.4 |

Planning latency: median 320 ms, p95 441 ms over 51 plans.

## Recovery after an obstacle moves mid-instruction

Detected and invalidated: 10/10; replanned on the new layout version: 10/10; human actions taken on a stale plan: 0; target later hit: 9/10.

## Occluded camera

Clear view requested: True; resumed after view restored: True; model update skipped for the trial with a tracking gap: True; outcome of the occluded roll: hit (observed) (an unobserved miss is reported as unknown, see test_core.py).

## Repeatability (first technical checkpoint)

8 Newton releases of one fixed setup with 1 mm gate jitter: exit point std 3.8 mm (x) / 0.9 mm (y); closest approach to target 125.6 ± 3.7 mm.

## OpenCV measurement accuracy (vs. renderer ground truth)

| Quantity | Mean | p95 | Samples |
|---|---|---|---|
| ball | 0.29 | 0.50 | 4592 |
| guide_mm | 0.15 | 0.32 | 4978 |
| guide_deg | 0.05 | 0.13 | 4978 |
| ms | 12.29 | 24.47 | 4978 |
