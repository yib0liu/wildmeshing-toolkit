#!/usr/bin/env python3
"""Check sweep outputs: did the run reach its target, is the output sane, and
does it still look like the input?

For every model in a sweep output directory this reports

    input V/F, output V/F, target V, reached target,
    non-manifold / boundary edge counts, bounding-box agreement

and can render a few side-by-side wireframes (input vs. output) as PNGs so the
simplification can actually be looked at.

Usage:
    python3 verify_results.py --run runs/qslim --run runs/shortest_edge_collapse \
        --out verification [--plot 6]
"""
from __future__ import annotations

import argparse
import csv
import json
import struct
from pathlib import Path

import numpy as np


def parse_stl(path: Path):
    """Binary STL (Thingi10K is 99.6% STL). Returns unique vertices as (n,3)."""
    data = path.read_bytes()
    if len(data) < 84:
        return np.zeros((0, 3))
    ntri = struct.unpack("<I", data[80:84])[0]
    expected = 84 + ntri * 50
    if expected != len(data):
        # ASCII STL or a truncated file: fall back to scanning "vertex" lines
        verts = []
        for line in data.decode("utf-8", "ignore").splitlines():
            parts = line.split()
            if len(parts) == 4 and parts[0] == "vertex":
                try:
                    verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
                except ValueError:
                    pass
        return np.unique(np.round(np.asarray(verts), 6), axis=0) if verts else np.zeros((0, 3))
    facets = np.frombuffer(data, dtype=np.uint8, offset=84, count=ntri * 50).reshape(ntri, 50)
    tris = np.frombuffer(facets[:, 12:48].tobytes(), dtype=np.float32).reshape(ntri, 3, 3)
    return np.unique(np.round(tris.reshape(-1, 3), 6), axis=0)


def parse_obj(path: Path):
    """OBJ writer used by WMTK. Returns (V, F) with 0-based indices."""
    verts, faces = [], []
    # a few Thingi10K .obj files are not valid UTF-8; do not let one bad byte
    # abort the whole verification
    with path.open(errors="replace") as handle:
        for line in handle:
            if line.startswith("v "):
                parts = line.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                idx = []
                for token in line.split()[1:]:
                    idx.append(int(token.split("/")[0]) - 1)
                if len(idx) >= 3:
                    faces.append(idx[:3])
    return np.asarray(verts, dtype=float), np.asarray(faces, dtype=np.int64)


def edge_valence(faces):
    """Edges seen once are boundary, seen 3+ times are non-manifold.

    Vectorised: a Python loop over faces is minutes per mesh on the larger
    Thingi10K models.
    """
    if len(faces) == 0:
        return 0, 0
    f = np.asarray(faces, dtype=np.int64)
    pairs = np.concatenate(
        [f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=0
    )
    pairs.sort(axis=1)
    keys = pairs[:, 0].astype(np.int64) * (int(pairs.max()) + 1) + pairs[:, 1]
    _, counts = np.unique(keys, return_counts=True)
    return int((counts == 1).sum()), int((counts > 2).sum())


def bbox_diag(verts):
    if len(verts) == 0:
        return 0.0
    return float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0)))


def target_from_log(model_dir: Path, fallback_vertices: int, default_rel: float = 0.1):
    """The components disagree on what they put in report.json: qslim writes
    target_vertices, shortest_edge_collapse does not -- recover the target from
    the run log ('target #V = N' / 'target number of verts: N'), else from the
    documented default target_rel."""
    import re

    for name in ("console.log", "run.log"):
        log = model_dir / name
        if not log.exists():
            continue
        text = log.read_text(errors="replace")
        for pattern in (r"target #V\s*=\s*([0-9]+)", r"target number of verts:\s*([0-9]+)"):
            hits = re.findall(pattern, text)
            if hits:
                return float(hits[0])
    return float(fallback_vertices) * default_rel


def analyse_run(run_dir: Path, sample=0):
    rows = []
    model_dirs = sorted(d for d in (run_dir / "success").glob("*") if d.is_dir())
    if sample and len(model_dirs) > sample:
        rng = np.random.default_rng(0)
        model_dirs = [model_dirs[i] for i in sorted(rng.choice(len(model_dirs), sample, replace=False))]
    for model_dir in model_dirs:
        config = json.loads((model_dir / "config.json").read_text())
        report_file = model_dir / "report.json"
        report = json.loads(report_file.read_text()) if report_file.exists() else {}
        output = Path(config["output"])
        if not output.exists():
            # sweep writes the mesh next to config.json; fall back to any .obj
            candidates = list(model_dir.glob("*.obj"))
            if not candidates:
                rows.append({"id": model_dir.name, "app": config["application"], "error": "no output mesh"})
                continue
            output = candidates[0]

        out_v, out_f = parse_obj(output)
        in_path = Path(config["input"])
        if in_path.suffix.lower() == ".stl":
            in_v = parse_stl(in_path)
            in_f_count = -1
        else:
            in_v, in_f = parse_obj(in_path)
            in_f_count = len(in_f)

        wall_file = model_dir / "wall_sec"
        wall_sec = float(wall_file.read_text().strip()) if wall_file.exists() else ""
        target = report.get("target_vertices")
        if target in (None, ""):
            target = target_from_log(model_dir, len(in_v))
        reached = "" if target in (None, "") else bool(len(out_v) <= float(target))
        boundary, nonmanifold = edge_valence(out_f) if len(out_f) else (0, 0)
        diag_in, diag_out = bbox_diag(in_v), bbox_diag(out_v)
        rows.append({
            "id": model_dir.name,
            "app": config["application"],
            "in_V": len(in_v),
            "in_F": in_f_count,
            "out_V": len(out_v),
            "out_F": len(out_f),
            "target_V": target,
            "reached_target": reached,
            "boundary_edges": boundary,
            "nonmanifold_edges": nonmanifold,
            "bbox_diag_in": round(diag_in, 6),
            "bbox_diag_out": round(diag_out, 6),
            "bbox_ratio": round(diag_out / diag_in, 4) if diag_in else "",
            # SEC writes no report file at all (its spec has no `report` key):
            # fall back to the wall-clock the sweep recorded
            "time_sec": report.get("time_sec", wall_sec),
        })
    return rows


def plot_examples(rows, runs, plot_dir: Path, n: int):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except ImportError:
        print("matplotlib unavailable; skipping plots")
        return []

    by_app = {}
    for run_dir in runs:
        for model_dir in sorted((run_dir / "success").glob("*")):
            if model_dir.is_dir():
                by_app.setdefault((model_dir.name, run_dir.name), model_dir)

    # pick the largest inputs so the simplification is actually visible
    row_v = {r["id"]: r.get("in_V", 0) for r in rows}
    candidates = sorted((name for name, _ in by_app if name in row_v),
                        key=lambda k: -row_v[k])[:n]
    plot_dir.mkdir(parents=True, exist_ok=True)
    made = []
    for name in candidates:
        panels = [(run_dir.name, by_app[(name, run_dir.name)]) for run_dir in runs
                  if (name, run_dir.name) in by_app]
        in_path = None
        for _, model_dir in panels:
            cfg = json.loads((model_dir / "config.json").read_text())
            in_path = Path(cfg["input"])
            break
        if in_path is None:
            continue
        in_v = parse_stl(in_path) if in_path.suffix.lower() == ".stl" else parse_obj(in_path)[0]

        fig = plt.figure(figsize=(5 * (len(panels) + 1), 5))
        ax = fig.add_subplot(1, len(panels) + 1, 1, projection="3d")
        sample = in_v[:: max(1, len(in_v) // 4000)]
        ax.scatter(sample[:, 0], sample[:, 1], sample[:, 2], s=1)
        ax.set_title(f"input\n{name} ({len(in_v)}v)")
        for i, (label, model_dir) in enumerate(panels, start=2):
            cfg = json.loads((model_dir / "config.json").read_text())
            out = Path(cfg["output"])
            if not out.exists():
                cand = list(model_dir.glob("*.obj"))
                if not cand:
                    continue
                out = cand[0]
            v, f = parse_obj(out)
            ax = fig.add_subplot(1, len(panels) + 1, i, projection="3d")
            if len(f):
                step = max(1, len(f) // 2500)
                mesh = Poly3DCollection(v[f[::step]], facecolors="cyan", linewidths=0.2,
                                        edgecolors="k", alpha=0.6)
                ax.add_collection3d(mesh)
                ax.scatter(v[:, 0], v[:, 1], v[:, 2], s=0.2, alpha=0.0)
                ax.set_xlim(v[:, 0].min(), v[:, 0].max())
                ax.set_ylim(v[:, 1].min(), v[:, 1].max())
                ax.set_zlim(v[:, 2].min(), v[:, 2].max())
            ax.set_title(f"{label}\n{len(v)}v / {len(f)}f")
        for ax in fig.axes:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
        path = plot_dir / f"{name}.png"
        fig.savefig(path, dpi=90, bbox_inches="tight")
        plt.close(fig)
        made.append(path)
    return made


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, action="append", required=True,
                        help="sweep output dir, e.g. runs/qslim (repeatable)")
    parser.add_argument("--out", type=Path, default=Path("verification"))
    parser.add_argument("--plot", type=int, default=0, help="render N example comparisons")
    parser.add_argument("--sample", type=int, default=0,
                        help="verify only N randomly chosen models per run (0 = all; seed 0)")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for run_dir in args.run:
        rows.extend(analyse_run(run_dir, args.sample))

    columns = ["id", "app", "in_V", "in_F", "out_V", "out_F", "target_V", "reached_target",
               "boundary_edges", "nonmanifold_edges", "bbox_diag_in", "bbox_diag_out",
               "bbox_ratio", "time_sec"]
    with (args.out / "verification.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    reached = [r for r in rows if r.get("reached_target") is True]
    missed = [r for r in rows if r.get("reached_target") is False]
    with_nm = [r for r in rows if isinstance(r.get("nonmanifold_edges"), int) and r["nonmanifold_edges"] > 0]
    ratios = [r["bbox_ratio"] for r in rows if isinstance(r.get("bbox_ratio"), float)]
    shrink = [r for r in rows if isinstance(r.get("bbox_ratio"), float) and r["bbox_ratio"] < 0.9]

    lines = [
        "# Verification",
        "",
        f"- runs inspected: {', '.join(str(p) for p in args.run)}",
        f"- models: {len(rows)}",
        f"- reached target #V: {len(reached)}; stopped above target: {len(missed)}",
        f"- outputs with non-manifold edges: {len(with_nm)}",
    ]
    if ratios:
        lines.append(f"- bbox diagonal ratio out/in: median {sorted(ratios)[len(ratios)//2]:.3f}, "
                     f"min {min(ratios):.3f}, max {max(ratios):.3f}")
        lines.append(f"- outputs shrinking more than 10% in bbox: {len(shrink)}")
    if missed:
        lines += ["", "## Stopped above the target (top 20 by gap)", "",
                  "| id | app | out_V | target_V | gap |", "|---|---|---:|---:|---:|"]
        for r in sorted(missed, key=lambda r: -(r["out_V"] - float(r["target_V"] or 0)))[:20]:
            lines.append(f"| {r['id']} | {r['app']} | {r['out_V']} | {r['target_V']} | "
                         f"{r['out_V'] - float(r['target_V'] or 0):.0f} |")
    if args.plot:
        made = plot_examples(rows, args.run, args.out / "plots", args.plot)
        lines += ["", f"## Plots ({len(made)})", ""]
        lines += [f"- {p.name}: {p}" for p in made]
    (args.out / "verification.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
