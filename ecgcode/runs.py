"""Run folders: build (anonymized record files + requests), results I/O, answer validation, parse.

A run folder holds:
  manifest.json      config, prompt sha256, split, record ids, git state, freeze info
  key.json           anonymous id -> PTB-XL ecg_id (never sent to the API)
  files/<anon>.csv   the raw recording of each record (uploaded to the Files API at submit time)
  requests.jsonl     request parameters per record, with a file-id placeholder
  files.json         anonymous id -> uploaded file id (written by `upload`)
  results.jsonl      one line per attempt: final answer, stop reason, usage summed over calls,
                     the code the model ran and its outputs
  predictions.csv, failures.csv, usage.json   written by parse
"""
import contextlib
import datetime as dt
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from .config import PROJECT_ROOT, config_diff, load_config, project_path
from .data import (LABELS, TEST_FOLD, TRAIN_FOLDS, VAL_FOLD, benchmark_split, class_priors,
                   load_metadata, load_signal)
from .prompting import load_system_prompt, request_params
from .records import record_csv

SPLITS = {"train": TRAIN_FOLDS, "dev": VAL_FOLD, "test": TEST_FOLD}
PROB_COLS = [f"p_{label}" for label in LABELS]
PROTECTED_PATHS = ["ecgcode", "config", "prompts", "requirements.txt"]
FREEZE_TAG = "protocol-frozen"


# --- provenance and freeze ---------------------------------------------------------

def _git(*args):
    return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True)


def git_state():
    try:
        top = _git("rev-parse", "--show-toplevel")
        if top.returncode != 0 or Path(top.stdout.strip()).resolve() != PROJECT_ROOT.resolve():
            return {"commit": None, "dirty": None, "dirty_files": [], "note": "not the repository root"}
        dirty = [l[3:] for l in _git("status", "--porcelain", "--", *PROTECTED_PATHS).stdout.splitlines()]
        return {"commit": _git("rev-parse", "HEAD").stdout.strip() or None, "dirty": bool(dirty),
                "dirty_files": dirty}
    except OSError:
        return {"commit": None, "dirty": None, "dirty_files": [], "note": "git not available"}


def check_frozen(prompt_sha):
    """Reasons why a test-fold build is not allowed yet (empty list = frozen correctly)."""
    problems, state = [], git_state()
    if state["commit"] is None:
        problems.append("the code is not in its own committed git repository")
    elif state["dirty"]:
        problems.append(f"uncommitted changes in {', '.join(state['dirty_files'][:5])}")
    if _git("rev-parse", "-q", "--verify", f"refs/tags/{FREEZE_TAG}").returncode != 0:
        problems.append(f"git tag '{FREEZE_TAG}' does not exist (see PROTOCOL.md)")
    elif _git("diff", "--quiet", FREEZE_TAG, "--", *PROTECTED_PATHS).returncode != 0:
        problems.append(f"code, config or prompts changed since tag '{FREEZE_TAG}'")
    protocol = project_path("PROTOCOL.md")
    if not protocol.exists() or prompt_sha not in protocol.read_text():
        problems.append("the prompt's sha256 is not recorded in PROTOCOL.md")
    return problems


def experiment_changes(cfg, base=None):
    base = base or load_config()
    return sorted((s, k) for s, values in cfg.items() if s not in ("budget", "cost")
                  for k, v in values.items() if base.get(s, {}).get(k, "<missing>") != v)


@contextlib.contextmanager
def run_lock(run_dir):
    """Only one money-related command per run at a time."""
    try:
        import fcntl
    except ImportError:
        yield
        return
    with open(Path(run_dir) / ".lock", "w") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"{run_dir}: another command is working on this run; let it finish first.")
        yield


def read_ids(path):
    return [int(line.split()[0]) for line in Path(path).read_text().splitlines()
            if line.strip() and not line.startswith("#")]


# --- build -------------------------------------------------------------------------

def build_run(cfg, run_dir, split, n=None, sample_seed=0, include_ids=None, exclude_ids=None,
              allow_unfrozen=False):
    run_dir = Path(run_dir)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"{run_dir} exists and is not empty; use a new run folder")
    system_text, prompt_sha = load_system_prompt(cfg)
    if split == "test" and not allow_unfrozen:
        problems = check_frozen(prompt_sha)
        changed = [f"{s}.{k}" for s, k in experiment_changes(cfg) if (s, k) != ("run", "seed")]
        if changed:
            problems.append(f"settings differ from the frozen config: {', '.join(changed)}")
        if problems:
            raise SystemExit("Refused to build a test-fold run; the protocol is not frozen:\n  - "
                             + "\n  - ".join(problems))
    meta = load_metadata(project_path(cfg["data"]["ptbxl_dir"]))
    df = benchmark_split(meta, SPLITS[split])
    if include_ids:
        df = df.loc[[i for i in dict.fromkeys(include_ids) if i in df.index]]
    if exclude_ids:
        df = df.drop(index=[i for i in set(exclude_ids) if i in df.index])
    if n:
        df = df.sample(n=min(n, len(df)), random_state=sample_seed)
    ecg_ids = sorted(int(i) for i in df.index)
    order = ecg_ids[:]
    random.Random(cfg["run"]["seed"]).shuffle(order)
    prefix = {"train": "t", "dev": "d", "test": "e"}[split]
    key = {f"{prefix}{k + 1:04d}": eid for k, eid in enumerate(order)}

    tmp = run_dir.parent / f".{run_dir.name}.building"
    if tmp.exists():
        shutil.rmtree(tmp)
    (tmp / "files").mkdir(parents=True)
    created = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    run_tag = hashlib.sha256(f"{run_dir.resolve()}|{created}".encode()).hexdigest()[:8]
    fs = int(cfg["input"]["sampling_rate"])
    example_files = _write_examples(cfg, meta, tmp / "examples", set(ecg_ids))
    with open(tmp / "requests.jsonl", "w") as f:
        for k, (anon, eid) in enumerate(key.items(), 1):
            csv = record_csv(load_signal(meta, eid, fs), fs, int(cfg["input"]["decimals"]))
            (tmp / "files" / f"{anon}.csv").write_text(csv)
            age, sex = meta.loc[eid, "age"], meta.loc[eid, "sex"]
            age = None if pd.isna(age) or age >= 300 else float(age)
            f.write(json.dumps({"custom_id": f"{run_tag}-req{k:05d}", "record_ids": [anon],
                                "params": request_params(cfg, system_text, anon, age, sex,
                                                         example_files)}) + "\n")
    (tmp / "key.json").write_text(json.dumps(key, indent=0))
    state = git_state()
    manifest = {
        "created_utc": created, "run_tag": run_tag, "split": split, "folds": SPLITS[split],
        "n_records": len(key), "ecg_ids": ecg_ids,
        "sample": {"n": n, "sample_seed": sample_seed, "include_ids": bool(include_ids),
                   "excluded_ids": len(set(exclude_ids or []))},
        "prompt_file": cfg["prompt"]["system"], "prompt_sha256": prompt_sha,
        "config": cfg, "config_diff_from_default": config_diff(cfg), "git": state,
        "frozen_build": split == "test" and not allow_unfrozen,
        "priors_train_folds": class_priors(meta),
    }
    (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if run_dir.exists():
        run_dir.rmdir()
    tmp.rename(run_dir)
    return manifest


def _write_examples(cfg, meta, folder, run_ids):
    """Labelled training-fold examples: n_per_class records per superclass, preferring records with
    that superclass only; written as exNN.csv plus example_labels.csv. Returns the file names."""
    k = int(cfg["examples"]["n_per_class"])
    if k <= 0:
        return []
    train = benchmark_split(meta, TRAIN_FOLDS).drop(index=[i for i in run_ids if i in meta.index], errors="ignore")
    chosen = []
    for label in LABELS:
        pool = train[(train[f"y_{label}"] == 1) & ~train.index.isin(chosen)]
        single = pool[pool["labels"].str.len() == 1]
        pick = (single if len(single) >= k else pool).sample(n=k, random_state=int(cfg["examples"]["seed"]))
        chosen += [int(i) for i in pick.index]
    order = chosen[:]
    random.Random(int(cfg["examples"]["seed"])).shuffle(order)
    folder.mkdir(parents=True)
    fs, rows, names = int(cfg["input"]["sampling_rate"]), [], []
    statements = pd.read_csv(Path(meta.attrs["root"]) / "scp_statements.csv", index_col=0)
    for n, eid in enumerate(order, 1):
        name = f"ex{n:02d}.csv"
        (folder / name).write_text(record_csv(load_signal(meta, eid, fs), fs, int(cfg["input"]["decimals"])))
        codes = meta.loc[eid, "scp_codes"]
        desc = "; ".join(f"{statements.loc[c, 'description']} ({int(v)}%)" if c in statements.index else c
                         for c, v in sorted(codes.items(), key=lambda kv: -kv[1]))
        age = meta.loc[eid, "age"]
        rows.append({"file": name, "age": "90+" if age >= 90 else int(age),
                     "sex": "female" if int(meta.loc[eid, "sex"]) == 1 else "male",
                     "superclasses": " ".join(meta.loc[eid, "labels"]), "statements": desc})
        names.append(name)
    pd.DataFrame(rows).to_csv(folder / "example_labels.csv", index=False)
    (folder / "example_ids.json").write_text(json.dumps(dict(zip(names, order))))
    return names + ["example_labels.csv"]


# --- loading and results I/O ----------------------------------------------------------

def load_manifest(run_dir):
    return json.loads((Path(run_dir) / "manifest.json").read_text())


def load_requests(run_dir):
    with open(Path(run_dir) / "requests.jsonl") as f:
        return [json.loads(line) for line in f]


def load_results(run_dir):
    path = Path(run_dir) / "results.jsonl"
    if not path.exists():
        return []
    out, bad = [], 0
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
    if bad:
        print(f"Warning: skipped {bad} malformed line(s) in {path}")
    return out


def append_results(run_dir, records):
    if not records:
        return
    path = Path(run_dir) / "results.jsonl"
    prefix = ""
    if path.exists() and path.stat().st_size:
        with open(path, "rb") as f:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                prefix = "\n"
    with open(path, "a") as f:
        f.write(prefix + "".join(json.dumps(r) + "\n" for r in records))
        f.flush()
        os.fsync(f.fileno())


# --- answer validation -------------------------------------------------------------------

def _norm_id(text):
    return re.sub(r"\s+", "", str(text)).lower()


def _extract_json(text):
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        match = re.search(r"\{.*\}", str(text or ""), flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
    return None


def validate_answer(res, record_ids, aggregate_how=None):
    """(predictions {anon: {...}}, failures {anon: reason}, n_clipped) for one result. With
    aggregate_how, the p_* columns come from the statement probabilities and the model's own
    superclass probabilities are kept as direct_p_*."""
    if res.get("type") != "succeeded":
        return {}, {a: f"api_{res.get('type')}" for a in record_ids}, 0
    stop = res.get("stop_reason")
    if stop != "end_turn":
        return {}, {a: f"stop_{stop}" for a in record_ids}, 0
    parsed = _extract_json(res.get("text"))
    entries = parsed.get("records") if isinstance(parsed, dict) else None
    if isinstance(parsed, dict) and entries is None and "p_NORM" in parsed:
        entries = [parsed]  # a single bare record object
    if not isinstance(entries, list):
        return {}, {a: "unparseable_json" for a in record_ids}, 0
    by_id = {}
    for e in entries:
        if isinstance(e, dict) and "record_id" in e:
            by_id.setdefault(_norm_id(e["record_id"]), e)
    if len(record_ids) == 1 and len(entries) == 1 and isinstance(entries[0], dict):
        by_id.setdefault(_norm_id(record_ids[0]), entries[0])
    preds, failures, clipped = {}, {}, 0
    for anon in record_ids:
        entry = by_id.get(_norm_id(anon))
        if entry is None:
            failures[anon] = "record_missing_in_response"
            continue
        values = {}
        for col in PROB_COLS:
            v = entry.get(col)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                values = None
                break
            clipped += v < 0 or v > 1
            values[col] = float(min(1.0, max(0.0, v)))
        if values is None:
            failures[anon] = "invalid_probability"
            continue
        pred = {**values, "key_findings": str(entry.get("key_findings", ""))}
        if aggregate_how:
            from .statements import aggregate
            stm = entry.get("statements")
            if not isinstance(stm, dict) or not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                                                    and math.isfinite(v) for v in stm.values()):
                failures[anon] = "invalid_statements"
                continue
            pred.update({f"direct_{c}": values[c] for c in PROB_COLS})
            pred.update({f"p_{k}": v for k, v in aggregate(stm, aggregate_how).items()})
            pred["statements"] = json.dumps(stm)
        preds[anon] = pred
    return preds, failures, clipped


def request_status(run_dir):
    groups = {r["custom_id"]: r["record_ids"] for r in load_requests(run_dir)}
    status = {cid: {"attempts": 0, "answered": set(), "last_failure": None} for cid in groups}
    for res in load_results(run_dir):
        s = status.get(res.get("custom_id"))
        if s is None or res.get("type") in ("canceled", "expired"):
            continue
        s["attempts"] += 1
        preds, failures, _ = validate_answer(res, groups[res["custom_id"]])
        s["answered"] |= set(preds)
        if failures:
            s["last_failure"] = next(iter(failures.values()))
    for cid, s in status.items():
        s["complete"] = s["answered"] >= set(groups[cid])
    return status


# --- parse ---------------------------------------------------------------------------------

def parse_run(run_dir):
    """results.jsonl -> predictions.csv, failures.csv, usage.json (cost over all attempts)."""
    run_dir = Path(run_dir)
    manifest = load_manifest(run_dir)
    key = json.loads((run_dir / "key.json").read_text())
    priors = manifest["priors_train_folds"]
    groups = {r["custom_id"]: r["record_ids"] for r in load_requests(run_dir)}
    out_cfg = manifest["config"].get("output", {})
    how = out_cfg.get("aggregate") if out_cfg.get("statement_level") else None
    sent = set()
    preds, failures, clipped, usd, usage = {}, {}, 0, 0.0, {}
    tool_runs, calls, attempts = [], [], 0
    for res in load_results(run_dir):
        cid = res.get("custom_id")
        if cid not in groups:
            continue
        sent.add(cid)
        attempts += 1
        usd += float(res.get("usd") or 0.0)
        for k, v in (res.get("usage") or {}).items():
            if isinstance(v, (int, float)):
                usage[k] = usage.get(k, 0) + v
        if res.get("type") == "succeeded":
            tool_runs.append(res.get("code_runs", 0))
            calls.append(res.get("n_calls", 1))
        p, f, c = validate_answer(res, groups[cid], how)
        clipped += c
        preds.update(p)
        for anon, reason in f.items():
            if anon not in preds:
                failures[anon] = reason
    for cid, record_ids in groups.items():
        for anon in record_ids:
            if anon not in preds and anon not in failures:
                failures[anon] = "no_result" if cid in sent else "not_sent"
    failures = {a: r for a, r in failures.items() if a not in preds}
    rows = []
    for anon, eid in key.items():
        if anon in preds:
            rows.append({"anon_id": anon, "ecg_id": eid, **preds[anon], "source": "model"})
        else:
            rows.append({"anon_id": anon, "ecg_id": eid, **{f"p_{l}": priors[l] for l in LABELS},
                         "key_findings": "", "source": "prior_fill"})
    pd.DataFrame(rows).sort_values("ecg_id").to_csv(run_dir / "predictions.csv", index=False)
    pd.DataFrame([{"anon_id": a, "ecg_id": key[a], "reason": r} for a, r in failures.items()],
                 columns=["anon_id", "ecg_id", "reason"]).to_csv(run_dir / "failures.csv", index=False)
    summary = {"records": len(key), "records_with_model_prediction": len(preds),
               "records_prior_filled": len(key) - len(preds), "attempts": attempts,
               "usd_total_all_attempts": round(usd, 4), "usage": usage, "probabilities_clipped": clipped,
               "code_runs_per_answer_mean": round(sum(tool_runs) / len(tool_runs), 2) if tool_runs else None,
               "api_calls_per_answer_mean": round(sum(calls) / len(calls), 2) if calls else None,
               "failure_reasons": pd.Series(list(failures.values()), dtype=str).value_counts().to_dict()}
    (run_dir / "usage.json").write_text(json.dumps(summary, indent=2))
    return summary
