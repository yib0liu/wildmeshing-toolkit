#!/usr/bin/env python3
"""Batch-run a WMTK surface component (qslim / isotropic_remeshing /
shortest_edge_collapse) over a folder of meshes.

The contract mirrors components/tetwild/.../run_tetwild_sweep.py, minus the
Thingi10K- and kirby-specific machinery (systemd scopes, tmux):

  * run it with no further arguments to start (or resume) a sweep; models
    already in success/ or failure/ are skipped;
  * per model: write a component JSON, run wmtk_app in a scratch dir with a
    time (and optional memory) cap;
  * exit 0                  -> move the run into <OUT>/success/<id>/
    nonzero / timeout / OOM -> move the run into <OUT>/failure/<id>/
  * afterwards write summary.csv and report.md.

Configuration -- command line (see --help) or environment:
    WMTK_APP       path to the wmtk_app binary
                   (default <repo>/build/app/wmtk_app)
    WMTK_OUT       output directory

Examples:
    # shortest edge collapse over the meshes shipped with the test data
    python3 run_component_sweep.py --application shortest_edge_collapse \
        --input-dir data/models --out runs/sec --threads 8

    # qslim keeping 10% of the vertices, 4 models in flight
    python3 run_component_sweep.py --application qslim --input-dir data/models \
        --out runs/qslim --param target_rel=0.1 --parallel 4

    # isotropic remeshing with an explicit target edge length
    python3 run_component_sweep.py --application isotropic_remeshing \
        --input-dir data/models --out runs/iso --param length_rel=0.01
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import resource
import shutil
import signal
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent
DEFAULT_APP = REPO / "build" / "app" / "wmtk_app"

APPLICATIONS = ("qslim", "isotropic_remeshing", "shortest_edge_collapse")
SURFACE_SUFFIXES = (".obj", ".stl", ".ply", ".off")

# Keys worth putting in the summary table; anything else in a component report
# is still kept in the per-model report file.
REPORT_COLUMNS = (
    "time_sec",
    "vertices",
    "triangles",
    "#v",
    "#f",
    "target_vertices",
    "target_length",
    "avg_length",
    "min_length",
    "max_length",
    "avg_valence",
    "nonmanifold_edges",
    "nonmanifold_vertices",
)


def parse_params(pairs):
    """--param k=v with automatic int/float/bool coercion."""
    out = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--param expects k=v, got {pair!r}")
        key, raw = pair.split("=", 1)
        low = raw.lower()
        if low in ("true", "false"):
            value = low == "true"
        else:
            try:
                value = int(raw)
            except ValueError:
                try:
                    value = float(raw)
                except ValueError:
                    value = raw
        out[key] = value
    return out


def discover(input_dir: Path, patterns):
    meshes = []
    for pattern in patterns:
        meshes.extend(sorted(input_dir.glob(pattern)))
    # unique, keep a stable order
    seen, unique = set(), []
    for mesh in meshes:
        if mesh.is_file() and mesh.resolve() not in seen:
            seen.add(mesh.resolve())
            unique.append(mesh)
    return unique


def _limit_memory(gb: float):
    """preexec_fn: cap address space so a runaway model is killed, not the box."""
    if gb and gb > 0:
        resource.setrlimit(resource.RLIMIT_AS, (int(gb * 1024**3),) * 2)


def run_one(mesh: Path, args, app: Path, out_dir: Path, params):
    """Run one mesh. Returns (status, reason)."""
    model_id = mesh.stem
    work = out_dir / ".work" / model_id
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    output_mesh = work / f"{model_id}.out.obj"
    report_path = work / "report.json"
    config = {
        "application": args.application,
        "input": str(mesh.resolve()),
        "output": str(output_mesh),
        "num_threads": args.threads,
        "report": str(report_path),
        "log_file": str(work / "run.log"),
    }
    config.update(params)
    (work / "config.json").write_text(json.dumps(config, indent=2))

    cmd = [str(app), "-j", str(work / "config.json")]
    log = (work / "console.log").open("w")
    started = time.time()
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            preexec_fn=_limit_memory(args.mem_gb) if args.mem_gb else None,
        )
        try:
            rc = proc.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            return "failure", f"timeout after {args.timeout}s"
    except MemoryError:
        return "failure", "memory cap"
    except OSError as exc:
        return "failure", f"could not start wmtk_app: {exc}"
    finally:
        log.close()

    elapsed = time.time() - started
    (work / "wall_sec").write_text(f"{elapsed:.3f}\n")

    if rc == 0:
        return "success", ""
    # surface the component's own last error line: a bare signal number is not
    # enough to tell "input has no faces" apart from a real crash
    last_error = ""
    console = work / "console.log"
    if console.exists():
        for line in console.read_text(errors="replace").splitlines():
            if "[error]" in line or "terminate" in line.lower():
                last_error = line.split("[error]")[-1].strip()
    detail = f" ({last_error})" if last_error else ""
    if rc < 0:
        sig = -rc
        if sig in (signal.SIGKILL, signal.SIGSEGV):
            return "failure", f"killed by signal {sig} (likely OOM){detail}"
        return "failure", f"killed by signal {sig}{detail}"
    return "failure", f"exit {rc}{detail}"


def process(mesh, args, app, out_dir, params):
    model_id = mesh.stem
    status, reason = run_one(mesh, args, app, out_dir, params)
    src = out_dir / ".work" / model_id
    dest = out_dir / status / model_id
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.exists():
        shutil.move(str(src), str(dest))
    if reason:
        (dest / "reason").write_text(reason + "\n")
    print(f"[{status}] {model_id} {reason}", flush=True)
    return model_id, status, reason


def collect_rows(out_dir: Path):
    rows = []
    for status in ("success", "failure"):
        for model_dir in sorted((out_dir / status).glob("*")):
            if not model_dir.is_dir():
                continue
            row = {"id": model_dir.name, "status": status, "reason": ""}
            reason_file = model_dir / "reason"
            if reason_file.exists():
                row["reason"] = reason_file.read_text().strip()
            wall = model_dir / "wall_sec"
            if wall.exists():
                row["wall_sec"] = float(wall.read_text().strip())
            report_file = model_dir / "report.json"
            if report_file.exists():
                try:
                    report = json.loads(report_file.read_text())
                except json.JSONDecodeError:
                    report = {}
                for key in REPORT_COLUMNS:
                    if key in report:
                        row[key] = report[key]
            rows.append(row)
    return rows


def write_report(out_dir: Path, rows, args, params):
    csv_path = out_dir / "summary.csv"
    columns = ["id", "status", "reason", "wall_sec"] + list(REPORT_COLUMNS)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    if not rows:
        return

    ok = [r for r in rows if r["status"] == "success"]
    bad = [r for r in rows if r["status"] != "success"]
    times = [r["wall_sec"] for r in ok if "wall_sec" in r]

    lines = [
        f"# Sweep report: {args.application}",
        "",
        f"- input:      {args.input_dir}",
        f"- models:     {len(rows)} ({len(ok)} success, {len(bad)} failure)",
        f"- threads:    {args.threads} per model, {args.parallel} models in flight",
        f"- timeout:    {args.timeout}s"
        + (f", memory cap {args.mem_gb}GB" if args.mem_gb else ""),
        f"- parameters: {json.dumps(params) if params else 'component defaults'}",
    ]
    if times:
        lines += [
            "",
            f"- wall time: median {statistics.median(times):.2f}s, "
            f"mean {statistics.mean(times):.2f}s, max {max(times):.2f}s",
        ]
    if bad:
        lines += ["", "## Failures", ""]
        for row in bad:
            lines.append(f"- {row['id']}: {row['reason']}")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--application", required=True, choices=APPLICATIONS)
    parser.add_argument("--input-dir", type=Path, default=REPO / "data" / "models")
    parser.add_argument("--pattern", action="append", default=None,
                        help=f"glob for input meshes (default: {', '.join('*' + s for s in SURFACE_SUFFIXES)})")
    parser.add_argument("--out", type=Path, default=None, help="output dir (default runs/<application>)")
    parser.add_argument("--wmtk-app", type=Path, default=None, help="path to wmtk_app")
    parser.add_argument("--threads", type=int, default=8, help="threads per model")
    parser.add_argument("--parallel", type=int, default=4, help="models run concurrently")
    parser.add_argument("--timeout", type=int, default=1800, help="per-model seconds")
    parser.add_argument("--mem-gb", type=float, default=0, help="per-model memory cap, 0 disables")
    parser.add_argument("--limit", type=int, default=0, help="process at most N new models (0 = all)")
    parser.add_argument("--param", action="append", default=[],
                        help="extra JSON spec entry, e.g. --param target_rel=0.1 (repeatable)")
    parser.add_argument("--report-only", action="store_true", help="only regenerate summary/report")
    args = parser.parse_args()

    app = Path(args.wmtk_app or os.environ.get("WMTK_APP") or DEFAULT_APP)
    if not app.is_file() and not args.report_only:
        sys.exit(f"wmtk_app not found: {app} (build it, or pass --wmtk-app)")

    out_dir = Path(args.out or os.environ.get("WMTK_OUT") or REPO / "runs" / args.application)
    out_dir.mkdir(parents=True, exist_ok=True)
    params = parse_params(args.param)
    patterns = args.pattern or [f"*{s}" for s in SURFACE_SUFFIXES]

    if args.report_only:
        write_report(out_dir, collect_rows(out_dir), args, params)
        print(f"report written to {out_dir / 'report.md'}")
        return

    meshes = discover(args.input_dir, patterns)
    if not meshes:
        sys.exit(f"no meshes matching {patterns} in {args.input_dir}")

    pending = [m for m in meshes
               if not (out_dir / "success" / m.stem).exists()
               and not (out_dir / "failure" / m.stem).exists()]
    todo = pending[: args.limit] if args.limit else pending
    deferred = len(pending) - len(todo)
    already = len(meshes) - len(pending)
    print(f"{len(meshes)} meshes found: {already} already done, {len(todo)} to process, "
          f"{deferred} deferred by --limit; application={args.application}", flush=True)

    if todo:
        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            list(pool.map(lambda m: process(m, args, app, out_dir, params), todo))

    rows = collect_rows(out_dir)
    write_report(out_dir, rows, args, params)
    ok = sum(1 for r in rows if r["status"] == "success")
    print(f"done: {ok}/{len(rows)} success -> {out_dir}")


if __name__ == "__main__":
    main()
