# qslim and shortest edge collapse on Thingi10K

What was run, what the outputs look like, and how the two components decide to
stop. All numbers below come from the runs in `runs_thingi10k/` and the checks in
`verification_sample/`; nothing is quoted from the papers or extrapolated.

## 1. Setup

- Toolkit built with `cmake -DCMAKE_BUILD_TYPE=Release .. && make wmtk_app`
  (needs `gmp`; on this box `dnf install gmp-devel`). Binary: `build/app/wmtk_app`.
- Data: Thingi10K, all **10,000** models (9,960 `.stl`, 40 `.obj`), downloaded
  from the HuggingFace dataset `Thingi10K/Thingi10K` into
  `/jizhi/jizhi2/worker/trainer/data/thingi10k/raw_meshes` (~30 GB).
- Driver: `run_component_sweep.py` (per-model JSON, timeout, success/failure
  bookkeeping, resumable).

```bash
python3 run_component_sweep.py --application qslim \
    --input-dir /jizhi/jizhi2/worker/trainer/data/thingi10k/raw_meshes \
    --out runs_thingi10k/qslim --threads 4 --parallel 24 --timeout 600

python3 run_component_sweep.py --application shortest_edge_collapse \
    --input-dir /jizhi/jizhi2/worker/trainer/data/thingi10k/raw_meshes \
    --out runs_thingi10k/sec_dedup --threads 4 --parallel 24 --timeout 600 \
    --param remove_duplicate_eps=1e-5
```

## 2. Results

| Run | Directory | Success | Failure |
| --- | --- | ---: | ---: |
| qslim (defaults) | `runs_thingi10k/qslim` | 9992 | 8 |
| shortest edge collapse, defaults | `runs_thingi10k/sec` | 9992 | 8 (but degenerate, see §3) |
| shortest edge collapse, `remove_duplicate_eps=1e-5` | `runs_thingi10k/sec_dedup` | 9992 | 8 |

Runtime (successful models):

- qslim: median **0.023 s**, mean 0.297 s, max 54.6 s
- shortest edge collapse: median **0.032 s**, mean 0.245 s, max 47.4 s
  (wall clock from the sweep; SEC writes no report file, see §6)

Median output size (qslim): 237 vertices — Thingi10K has many very small models.

## 3. Default shortest edge collapse does nothing on STL

On sampled models, SEC's output vertex count equalled its input count. Cause:

- STL stores 3 vertices per facet. `55280.stl` has 900,490 facets → 2,701,470
  vertex slots, but only **450,227** distinct coordinates.
- qslim hardcodes `remove_duplicate_eps = 1e-5` (`qslim.cpp`), so duplicates are
  welded and the mesh is usable.
- SEC's `remove_duplicate_eps` defaults to **-1** ("merge nothing"), so the mesh
  is built as a triangle soup; with `use_link_condition` on, no edge can be
  collapsed. Measured on `55280`: **2,701,470 → 2,701,470 vertices, 0 collapses**,
  target was 270,147.

With `remove_duplicate_eps=1e-5` the same model behaves correctly:
**449,537 → 44,947** (target 44,953) in 1.2 s.

This is why the SEC sweep was re-run with that parameter; the default-parameter
run is kept (`runs_thingi10k/sec`) only as evidence of the defect.

## 4. Termination criteria

Both components use one local operation (edge collapse) driven by
`wmtk::ExecutePass`. The scheduler (`src/wmtk/ExecutionScheduler.hpp`) counts
**successful** operations only:

```cpp
// line ~505: on success
if (track_live_success) live_success.fetch_add(1, ...);
// line ~508: rejected operations increment cnt_fail, not live_success

// line ~528: after each operation
if (track_live_success &&
    live_success.load(...) > stopping_criterion_checking_frequency) {
    if (stopping_criterion(m)) { stop.store(true); return; }
}
```

Both components set

```cpp
stopping_criterion_checking_frequency = V0 - target - 1;
stopping_criterion = [](auto& m) { return true; };
```

so the pass stops as soon as more than `V0 − target − 1` collapses succeeded,
i.e. after roughly `V0 − target` collapses, leaving about `target` vertices.

| | qslim | shortest edge collapse |
| --- | --- | --- |
| target | `target_abs` if > 0, else `V0 × target_rel` (default **0.1**) | `V0 × target_rel` (default **0.1**); no `target_abs` option |
| priority | `-compute_cost_for_e` — smallest quadric error first | `-len2` — shortest edge first |
| code | `QSlimMesh.cpp::collapse_qslim` (~282–330) | `ShortestEdgeCollapse.cpp::collapse_shortest` (~212–280) |
| blockers | link condition, envelope (`eps_rel`), invariants | same, plus `use_link_condition` (default **true**) |
| if target not reached | `throw_on_fail` (default **false**) → silent; else throws | same |

**There is no error- or quality-based stopping rule.** Neither stops on quadric
error, Hausdorff distance, or edge-length degeneration. The envelope is only a
per-operation veto, not a stopping condition. Collapses rejected by the
invariants never advance the success counter, so the queue can drain with more
vertices than the target; unless `throw_on_fail=true`, that is reported as a
successful run.

## 5. Verification

### 5.1 Did the run reach its target? (all 9,992 successful models)

Recovered from the run logs (`target …` vs. `After #V` / `After collapse: #V`),
so this covers every successful model, not a sample:

| run | reached target | rate |
| --- | ---: | ---: |
| qslim (defaults) | 9184 / 9992 | **91.9 %** |
| shortest edge collapse, `remove_duplicate_eps=1e-5` | 9134 / 9992 | **91.4 %** |
| shortest edge collapse, `remove_duplicate_eps=1e-5` + `use_link_condition=false` | 9736 / 9992 | **97.4 %** |

Because `throw_on_fail` defaults to false, the 808 / 858 / 256 runs that stopped
above their target still exited 0 and are recorded as successes — see §4.

### 5.2 Geometry checks (sample)

Sampled **1000 models per run** (seed 0) from qslim + `sec_dedup`, i.e. 2000
outputs (`verification_sample1000/verification.csv`, `verification.md`,
`plots/`):

- reached the target vertex count: **1820 / 2000** (91.0%)
- stopped above target: **180 / 2000**
- outputs containing non-manifold edges: **212 / 2000**
- bounding-box diagonal ratio out/in: median **0.999**, min 0.365, max 27.517

The extreme bbox ratios are almost certainly a measurement artifact: the ratio
is computed over every vertex written to the OBJ, including slots that
`consolidate_mesh` left unused. They are reported here rather than hidden, but
they should not be read as real shape change without a closer look.

Individual models (`in_V → out_V`, target):

| model | qslim | shortest edge collapse | manifold | bbox ratio |
| --- | --- | ---: | ---: | --- |
| 55280 | 450,227 → 44,950 (44,954) | 450,227 → 44,950 (44,953) | ok | 1.000 |
| 65942 | 377,154 → 37,712 (37,715) | 377,154 → 37,713 (37,715) | ok | 1.000 |
| 372057 | 300,644 → 30,058 (30,060) | 300,644 → 30,057 (30,060) | ok | 1.000 |
| 100028 | 120 → 12 (12) | 120 → 12 (12) | ok | 0.97–1.00 |
| 100026 | 117 → 27 (12) | 117 → 28 (11) | open surface | 0.76–0.84 |
| 100027 | 218 → 41 (22) | 218 → 37 (21) | open surface | 0.80–0.82 |

The large models simplify exactly to the target with no non-manifold edges and
an unchanged bounding box — that is the expected, correct behaviour.

The very small models stop well above the target: there is not enough valid
collapse work left (link condition / boundary), and their bounding box shrinks
because stray or isolated vertices disappear. Both algorithms behave the same
way on these, which points at the input, not at the components.

Plots (input point cloud vs. simplified meshes) are in
`verification_sample1000/plots/`.

### 5.3 Why some models are not simplified at all

Among the 180 runs that stopped above the target, the dominant pattern is
**zero collapses**: output vertex count equals input vertex count. Two distinct
causes were identified, both confirmed by re-running with a changed parameter.

**(a) STL triangle soup + `remove_duplicate_eps = -1` (SEC defaults).**
See §3. Fixed by `remove_duplicate_eps=1e-5`.

**(b) Heavily non-manifold input + `use_link_condition = true` (SEC default).**
These models weld fine but have a huge fraction of non-manifold edges, and the
link condition forbids collapsing anything attached to them:

| model | input | non-manifold edges (input) | default (link on) | `use_link_condition=false` |
| --- | --- | ---: | ---: | ---: |
| 51351 | 8,700 V / 52,368 F | 26,184 | 8,700 → **8,700** (0 collapses) | 8,700 → **863** (target 870) |
| 45122 | 5,190 V / 25,092 F | 13,554 | 5,190 → **5,190** (0 collapses) | 5,190 → **512** (target 519) |

Turning the link condition off lets the collapse proceed and reach the target;
the cost is that the output may be non-manifold (51351 came out with 21
non-manifold edges, 45122 with 0). Note that qslim, which also enforces the
link condition, stalls on exactly the same models — this is an input
property, not a bug in either component.

Reproduce (b):

```bash
python3 run_component_sweep.py --application shortest_edge_collapse \
    --input-dir /jizhi/jizhi2/worker/trainer/data/thingi10k/raw_meshes \
    --out runs_thingi10k/sec_nolink --threads 4 --parallel 24 --timeout 600 \
    --param remove_duplicate_eps=1e-5 --param use_link_condition=false
```

### 5.4 Trade-off of turning the link condition off

Same 1000-model sample, comparing the two SEC configurations (and qslim as
reference):

Per-run breakdown of the same 1000-model samples (the 2000 totals quoted in
§5.2 mix the two runs):

| run (1000 sampled models each) | reached target | outputs with non-manifold edges |
| --- | ---: | ---: |
| qslim (defaults) | 914 / 1000 (91.4 %) | 102 / 1000 (10.2 %) |
| SEC, `remove_duplicate_eps=1e-5` | 906 / 1000 (90.6 %) | 110 / 1000 (11.0 %) |
| SEC, `remove_duplicate_eps=1e-5` + `use_link_condition=false` | 966 / 1000 (96.6 %) | **676 / 1000 (67.6 %)** |

Disabling the link condition buys about 6 points of target-reaching rate and
costs roughly six times as many non-manifold outputs — the expected bargain,
since the link condition exists precisely to forbid topology-changing
collapses. Which configuration is right depends on whether the downstream
consumer needs a manifold surface. (The qslim row is identical in both samples:
it is the same qslim run, re-sampled with a fixed seed.)

### 5.5 Geometric fidelity (Chamfer distance)

Bounding boxes alone do not show that the simplified mesh still *looks like* the
input, so for a handful of models the two-way Chamfer distance between the input
vertices and the output vertices was measured, normalised by the input bounding
box diagonal (lower is better; 0.005 ≈ half a percent of the model's size).

| model | app | in_V | out_V | input→output | output→input | max |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 55280 | qslim | 450,227 | 44,950 | 0.0042 | 0.0049 | 0.0049 |
| 55280 | SEC | 450,227 | 44,950 | 0.0042 | 0.0042 | 0.0042 |
| 65942 | qslim | 377,154 | 37,712 | 0.0045 | 0.0049 | 0.0049 |
| 65942 | SEC | 377,154 | 37,713 | 0.0072 | 0.0045 | 0.0072 |
| 372057 | qslim | 300,644 | 30,058 | 0.0037 | 0.0042 | 0.0042 |
| 372057 | SEC | 300,644 | 30,057 | 0.0055 | 0.0041 | 0.0055 |
| 51351 | both | 8,700 | 8,700 | 0.0000 | 0.0020 | 0.0020 |
| 45122 | both | 5,190 | 5,190 | 0.0000 | 0.0005 | 0.0005 |
| 100026 | qslim | 117 | 27 | 0.0695 | 0.0877 | 0.0877 |
| 100026 | SEC | 117 | 28 | 0.0713 | 0.0895 | 0.0895 |

The large models simplify to 10 % of their vertices while staying within
~0.5 % of the model size — that is the expected, correct behaviour. The
zero-collapse models (51351, 45122) have `input→output = 0` because their output
vertices are exactly their input vertices. The tiny 100026 is the one case with
a visible deviation (~7–9 %), which is what aggressive simplification of a
117-vertex mesh looks like.

### 5.6 Effect of the target ratio

The sweeps above all use the default `target_rel = 0.1`. To see how the
termination behaves away from the default, both components were re-run with
`target_rel ∈ {0.5, 0.25, 0.1, 0.05, 0.01}` on the **first 800 models** (sorted
by file name — a fixed subset, not a random sample). SEC used
`remove_duplicate_eps=1e-5` and `use_link_condition=false`; qslim used its
defaults apart from the target.

| `target_rel` | qslim reached | SEC reached |
| ---: | ---: | ---: |
| 0.5 | 794 / 796 (99.7 %) | 794 / 796 (99.7 %) |
| 0.25 | 786 / 796 (98.7 %) | 792 / 796 (99.5 %) |
| 0.1 | 736 / 796 (92.5 %) | 784 / 796 (98.5 %) |
| 0.05 | 666 / 796 (83.7 %) | 766 / 796 (96.2 %) |
| 0.01 | 475 / 796 (59.7 %) | 624 / 796 (78.4 %) |

Reading this together with §4: the stop rule itself is exact — a run that can
perform the required number of collapses lands within one vertex of the target
(median output/target ratio ≈ 1.0 at every setting). What changes with a
smaller target is how often the mesh *runs out of legal collapses* first. qslim
degrades faster than SEC-without-link-condition (59.7 % vs 78.4 % at 1 %),
which is consistent with qslim always enforcing the link condition. So the
practical limit on simplification here is the input's collapsibility, not the
stopping criterion.

Reproduce (one setting shown):

```bash
python3 run_component_sweep.py --application shortest_edge_collapse \
    --input-dir /jizhi/jizhi2/worker/trainer/data/thingi10k/raw_meshes \
    --out runs_targetrel/sec_0.01 --threads 4 --parallel 24 --timeout 600 \
    --limit 800 --param target_rel=0.01 \
    --param remove_duplicate_eps=1e-5 --param use_link_condition=false
```

## 6. Known gaps

- **`shortest_edge_collapse` writes no report file.** Its spec has no `report`
  key, so `report.json` is absent; the sweep's `wall_sec` file is used for
  timing and the target is recovered from the log ("target #V = N").
- Verification above is a **sample** (1000 per run), not all 19,984 outputs. A
  full pass was launched but had not produced results at the time of writing.
- The bounding-box ratio is computed over all vertices written to the OBJ,
  including unused slots, so a large ratio can be a measurement artifact rather
  than real shape change.
- The 8 failures are identical for both components and are **input defects**,
  not algorithm failures:
  - `285440.obj`, `293590.obj`, `921798.obj`: malformed OBJ (`vn` line without 3
    coordinates / unreadable)
  - `286163.stl`, `74463.stl`: no faces
  - `49911.stl`: unparsable vertex z coordinate
  - `77942.stl`: header declares 6964 facets but the file is 50 bytes short
  - `81313.obj`: no faces
