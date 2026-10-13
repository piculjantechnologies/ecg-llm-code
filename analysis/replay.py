"""Local replay of the model's code runs (post hoc; written after the test run).

Run from the repository root:  python analysis/replay.py [--sample N] [--out DIR]
Writes analysis/replay_summary.json, analysis/replay_runs.json and, for a full replay, paper/generated/replay.tex
(macros prefixed RP).
Re-executes every code run of a run folder, record by record and in the original order, in an offline sandbox
(bubblewrap: no network, a private /tmp kept across the runs of one record, the record's file under $INPUT_DIR as in
the API container), with a local Python 3.11 that has NumPy, SciPy and pandas; then compares each run's printed output
and exit status with what the API's sandbox returned. Needs bwrap and that Python (--python-prefix, by default the
conda environment "replay"). No API calls; the run folder is only read.

Each run is classed as identical (same stdout), whitespace (same after stripping trailing whitespace per line),
rounding (same text, and every number within one unit of its last printed digit or, for numbers printed at full
precision, within a relative 1e-9: floating-point rounding of a differently built library) or different. A run
ended with an error as defined in posthoc.py (a non-zero exit status, or an uncaught Python exception other than a
broken pipe, since a piped command returns the exit status of its last command); an error is reproduced when the
local run also ends with one, of the same exception type.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from common import GEN, OUT_JSON, TEST, custom_to_anon, thousands, write_json, write_macros
from posthoc import run_error

INPUT_ID = "replay"      # the API container used a random directory name under /files/input
TIMEOUT_S = 300          # per code run
REL_TOL = 1e-9           # floating-point rounding, for numbers printed at full precision
NUM = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")
EXC = re.compile(r"^(\w+(?:Error|Exception))\b", re.M)


def exception_type(stderr):
    found = EXC.findall(stderr or "")
    return found[-1] if found else None


def code_runs(result):
    """(command, stdout, return code, error, exception type) of each code run, in order."""
    uses = [b for b in result["content"] if b.get("type") == "server_tool_use"]
    outs = {b.get("tool_use_id"): b.get("content") or {} for b in result["content"] if "result" in b.get("type", "")}
    runs = []
    for u in uses:
        c = outs.get(u["id"], {})
        runs.append((u["input"]["command"], c.get("stdout", "") or "", c.get("return_code"), run_error(c),
                     exception_type(c.get("stderr"))))
    return runs


def sandbox_cmd(prefix, rec_dir, command):
    return ["bwrap", "--unshare-all", "--die-with-parent", "--new-session",
            "--ro-bind", "/usr", "/usr", "--symlink", "usr/bin", "/bin", "--symlink", "usr/lib", "/lib",
            "--symlink", "usr/lib64", "/lib64", "--ro-bind", "/etc", "/etc", "--ro-bind", prefix, prefix,
            "--proc", "/proc", "--dev", "/dev",
            "--bind", str(rec_dir / "tmp"), "/tmp", "--bind", str(rec_dir / "work"), "/work",
            "--dir", "/files/input", "--bind", str(rec_dir / "input"), f"/files/input/{INPUT_ID}",
            "--setenv", "INPUT_DIR", f"/files/input/{INPUT_ID}", "--setenv", "HOME", "/work",
            "--setenv", "PATH", f"{prefix}/bin:/usr/bin:/bin", "--setenv", "OMP_NUM_THREADS", "1",
            "--setenv", "OPENBLAS_NUM_THREADS", "1", "--setenv", "MKL_NUM_THREADS", "1",
            "--chdir", "/work", "bash", "-c", command]


def compare(orig, new):
    if orig == new:
        return "identical", 0, 0.0, 0.0
    strip = lambda s: "\n".join(l.rstrip() for l in s.strip("\n").split("\n"))  # noqa: E731
    if strip(orig) == strip(new):
        return "whitespace", 0, 0.0, 0.0
    to, tn = NUM.split(strip(orig)), NUM.split(strip(new))
    no, nn = NUM.findall(strip(orig)), NUM.findall(strip(new))
    if len(no) != len(nn) or [re.sub(r"\s+", " ", t) for t in to] != [re.sub(r"\s+", " ", t) for t in tn]:
        return "different", None, None, None
    n_diff, max_diff, max_rel = 0, 0.0, 0.0
    for a, b in zip(no, nn):
        if a == b:
            continue
        x, y = float(a), float(b)
        mant = a.lower().split("e")[0]
        unit = 10.0 ** -(len(mant.split(".")[1]) if "." in mant else 0)
        if "e" in a.lower():
            unit *= 10.0 ** int(a.lower().split("e")[1])
        n_diff += 1
        max_diff = max(max_diff, abs(x - y))
        if x == y:   # e.g. -0.00 and 0.00
            continue
        max_rel = max(max_rel, abs(x - y) / max(abs(x), abs(y)))
        if abs(x - y) > max(unit * 1.000001, REL_TOL * max(abs(x), abs(y))):
            return "different", n_diff, max_diff, max_rel
    return "rounding", n_diff, max_diff, max_rel


def replay_record(args):
    anon, csv, runs, prefix, work_root = args
    rec_dir = Path(tempfile.mkdtemp(prefix=f"{anon}-", dir=work_root))
    try:
        for d in ("tmp", "work", "input"):
            (rec_dir / d).mkdir()
        shutil.copy(csv, rec_dir / "input" / csv.name)
        rows = []
        for i, (command, out, rc, err, exc) in enumerate(runs, 1):
            try:
                p = subprocess.run(sandbox_cmd(prefix, rec_dir, command), capture_output=True, text=True,
                                   timeout=TIMEOUT_S, errors="replace")
                new_out, new_rc, new_err = p.stdout, p.returncode, p.stderr
            except subprocess.TimeoutExpired:
                new_out, new_rc, new_err = "", "timeout", ""
            cls, n_diff, max_diff, max_rel = compare(out, new_out)
            new_c = {"return_code": new_rc if new_rc != "timeout" else -1, "stderr": new_err}
            rows.append({"record": anon, "run": i, "n_runs": len(runs), "class": cls, "rc_api": rc, "rc_local": new_rc,
                         "error_api": err, "error_local": run_error(new_c), "exception_api": exc,
                         "exception_local": exception_type(new_err),
                         "numbers_differing": n_diff, "max_abs_diff": max_diff, "max_rel_diff": max_rel,
                         "stdout_api": out if cls == "different" else None,
                         "stdout_local": new_out if cls == "different" else None})
        return rows
    finally:
        shutil.rmtree(rec_dir, ignore_errors=True)


def local_versions(prefix):
    code = ("import sys,numpy,scipy,pandas,matplotlib;print(sys.version.split()[0],numpy.__version__,"
            "scipy.__version__,pandas.__version__,matplotlib.__version__)")
    v = subprocess.run([f"{prefix}/bin/python", "-c", code], capture_output=True, text=True).stdout.split()
    return dict(zip(["python", "numpy", "scipy", "pandas", "matplotlib"], v))


def sci(x):
    m, e = f"{x:.1e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def macros(sm):
    c, n = sm["runs_by_class"], sm["code_runs"]
    v = sm["versions"]
    return {"RPRecords": thousands(sm["records"]), "RPRuns": thousands(n),
            "RPIdentical": thousands(c["identical"]), "RPIdenticalPct": f"{100 * c['identical'] / n:.1f}",
            "RPRounding": thousands(c["rounding"]), "RPWhitespace": thousands(c["whitespace"]),
            "RPDifferent": thousands(c["different"]),
            "RPErrorStatusMatched": thousands(sm["runs_error_status_matched"]),
            "RPErrorsAPI": thousands(sm["api_errors"]), "RPErrorsReproduced": thousands(sm["api_errors_reproduced"]),
            "RPRecordsReproduced": thousands(sm["records_all_reproduced"]),
            "RPRecordsIdentical": thousands(sm["records_all_identical"]),
            "RPMaxRel": sci(sm["rounding_max_rel_diff"]), "RPMaxAbs": sci(sm["rounding_max_abs_diff"]),
            "RPPython": v["python"], "RPNumPy": v["numpy"], "RPSciPy": v["scipy"], "RPPandas": v["pandas"],
            "RPMatplotlib": v["matplotlib"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(TEST))
    ap.add_argument("--sample", type=int, default=0, help="replay a random sample of this many records (0: all)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--python-prefix", default=str(Path.home() / "anaconda3" / "envs" / "replay"))
    ap.add_argument("--out", default=str(OUT_JSON))
    a = ap.parse_args()
    run, out = Path(a.run), Path(a.out)
    c2a = custom_to_anon(run)
    jobs = []
    with open(run / "results.jsonl") as f:
        for line in f:
            r = json.loads(line)
            anon = c2a[r["custom_id"]]
            jobs.append((anon, run / "files" / f"{anon}.csv", code_runs(r), a.python_prefix, None))
    jobs.sort(key=lambda j: j[0])
    if a.sample:
        idx = np.random.default_rng(a.seed).choice(len(jobs), a.sample, replace=False)
        jobs = [jobs[i] for i in sorted(idx)]
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="replay-") as work_root:
        jobs = [j[:4] + (work_root,) for j in jobs]
        with ProcessPoolExecutor(a.workers) as ex:
            rows = [row for rec in ex.map(replay_record, jobs) for row in rec]
    classes = ["identical", "whitespace", "rounding", "different"]
    by_rec = {}
    for r in rows:
        by_rec.setdefault(r["record"], []).append(r)
    summary = {
        "run": run.name, "records": len(by_rec), "code_runs": len(rows),
        "python_env": Path(a.python_prefix).name,   # the environment's name, not its local path
        "runs_by_class": {c: sum(r["class"] == c for r in rows) for c in classes},
        "runs_exit_status_matched": sum(str(r["rc_api"]) == str(r["rc_local"]) for r in rows),
        "runs_error_status_matched": sum(r["error_api"] == r["error_local"] for r in rows),
        "api_errors": sum(r["error_api"] for r in rows),
        "api_errors_reproduced": sum(r["error_api"] and r["error_local"] and r["exception_api"] == r["exception_local"]
                                     for r in rows),
        "local_errors_not_in_api": sum(r["error_local"] and not r["error_api"] for r in rows),
        "records_all_identical": sum(all(r["class"] == "identical" for r in rs) for rs in by_rec.values()),
        "records_all_reproduced": sum(all(r["class"] != "different" for r in rs) for rs in by_rec.values()),
        "rounding_max_rel_diff": max([r["max_rel_diff"] for r in rows if r["class"] == "rounding"] or [0.0]),
        "rounding_max_abs_diff": max([r["max_abs_diff"] for r in rows if r["class"] == "rounding"] or [0.0]),
        "versions": local_versions(a.python_prefix),
    }
    write_json(summary, out / "replay_summary.json")
    if not a.sample:
        write_macros(GEN / "replay.tex", macros(summary), "python analysis/replay.py (POST HOC local replay)")
    write_json(rows, out / "replay_runs.json")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
