"""Money: token cost, a spend ledger across all runs, and the pre-submit budget gate.

Every paid API result is appended to the ledger (runs/spend_ledger.jsonl) when it is
recorded, with its usage and computed cost (one line per message id; re-recording the
same message is ignored). `submit` refuses to start if
- the planned cost (worst case, or with --budget-basis expected a calibrated expectation)
  exceeds the --max-usd given for that submission, or
- ledger total + worst case of batches still running + worst case of this submission
  would exceed the total cap (always on the worst case: the cap is the most you can lose,
  whatever happens). The cap is set with `spend --set-cap` (runs/budget_cap.json, next to
  the ledger); [budget] total_usd_cap in the config is only the starting value,
and then asks for confirmation.
"""
import contextlib
import datetime as dt
import json
import os
import sys
from pathlib import Path

from .config import project_path


def ledger_path():
    return Path(os.environ.get("ECGREAD_LEDGER") or project_path("runs/spend_ledger.jsonl"))


def cap_path():
    """The spending cap you set with `spend --set-cap`; lives next to the ledger (git-ignored),
    so raising it never touches the frozen config."""
    return ledger_path().with_name("budget_cap.json")


def total_cap(budget_cfg):
    path = cap_path()
    if path.exists():
        return float(json.loads(path.read_text())["total_usd_cap"])
    return float(budget_cfg["total_usd_cap"])


def set_cap(usd):
    if not usd > 0:
        raise ValueError("the cap must be a positive number of USD")
    path = cap_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"total_usd_cap": float(usd),
                                "set_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}))


def usage_cost_usd(usage, cost, mode):
    """Cost of one API result. Batch results get the batch discount; sync results do not."""
    usage = usage or {}
    usd = (int(usage.get("input_tokens") or 0) * cost["input"]
           + int(usage.get("cache_read_input_tokens") or 0) * cost["cache_read"]
           + int(usage.get("cache_creation_input_tokens") or 0) * cost["input"] * cost["cache_write_multiplier"]
           + int(usage.get("output_tokens") or 0) * cost["output"]) * 1e-6
    return usd * (cost["batch_discount"] if mode == "batch" else 1.0)


def _recorded_message_ids():
    return {e["message_id"] for e in ledger_entries() if e.get("message_id")}


def record_spend(run_dir, results, cost):
    """Append one ledger line per paid result (results that carry usage), skipping message
    ids that are already in the ledger (e.g. a collect repeated after a crash)."""
    seen = _recorded_message_ids()
    lines = []
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    for r in results:
        if not r.get("usage") or (r.get("message_id") and r["message_id"] in seen):
            continue
        seen.add(r.get("message_id"))
        mode = r.get("mode", "batch")
        lines.append(json.dumps({"utc": now, "run": str(Path(run_dir).resolve()), "custom_id": r.get("custom_id"),
                                 "message_id": r.get("message_id"), "mode": mode, "usage": r["usage"],
                                 "usd": r["usd"] if "usd" in r else round(usage_cost_usd(r["usage"], cost, mode), 6)})
                     + "\n")
    if lines:
        path = ledger_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write("".join(lines))
            f.flush()
            os.fsync(f.fileno())


def record_possible_spend(run_dir, custom_id, usd, note):
    """Ledger line for money that may have been spent without a usage report (a dropped sync
    stream, an abandoned batch): counted at its worst case so the cap stays an upper bound."""
    path = ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    line = {"utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "run": str(Path(run_dir).resolve()), "custom_id": custom_id, "message_id": None,
            "estimated": True, "note": note, "usd": round(float(usd), 6)}
    with open(path, "a") as f:
        f.write(json.dumps(line) + "\n")
        f.flush()
        os.fsync(f.fileno())


@contextlib.contextmanager
def submit_lock():
    """One submit at a time across all runs, so two budget checks cannot both pass a cap that
    only one of them fits."""
    try:
        import fcntl
    except ImportError:
        yield
        return
    path = ledger_path().with_name(".submit.lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another submit is running (possibly waiting for its confirmation); finish it first.")
        yield


def ledger_entries():
    path = ledger_path()
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _unique_entries():
    seen, out = set(), []
    for e in ledger_entries():
        mid = e.get("message_id")
        if mid and mid in seen:
            continue
        seen.add(mid)
        out.append(e)
    return out


def spent_total():
    return sum(e.get("usd", 0.0) for e in _unique_entries())


def spent_by_run():
    totals = {}
    for e in _unique_entries():
        totals[e["run"]] = totals.get(e["run"], 0.0) + e.get("usd", 0.0)
    return totals


def check_budget(plan, budget_cfg, max_usd, basis="worst", committed=0.0):
    """Refuse (SystemExit) when the planned spend exceeds --max-usd or the global cap.
    `committed` is the worst case of batches submitted earlier and not yet collected."""
    if basis == "expected" and not plan.get("calibration"):
        raise SystemExit("Refused: --budget-basis expected needs measured output tokens, and there are none "
                         "for this model, effort and records per request yet. Use the worst-case basis, or "
                         "send a few requests first (--limit 10) and collect them.")
    planned = plan["usd_worst_case"] if basis == "worst" else plan["usd_expected"]
    label = "worst case (every request uses max_tokens)" if basis == "worst" else "expected"
    if planned > max_usd:
        raise SystemExit(f"Refused: {label} cost ${planned:.2f} exceeds --max-usd {max_usd:.2f}. "
                         f"Raise --max-usd deliberately, submit fewer requests (--limit), or use "
                         f"--budget-basis expected once a pilot has calibrated the estimate.")
    spent = spent_total()
    cap = total_cap(budget_cfg)
    worst = plan["usd_worst_case"]
    if spent + committed + worst > cap:
        raise SystemExit(f"Refused: ledger total ${spent:.2f} + running batches (worst case) ${committed:.2f} "
                         f"+ this submission's worst case ${worst:.2f} = ${spent + committed + worst:.2f} would "
                         f"exceed your total cap ${cap:.2f}. Raise it deliberately if this stage is intended: "
                         f"python -m ecgread spend --set-cap USD")
    return planned, spent


def confirm(summary_lines, token, assume_yes=False):
    """Print what will be spent and require the user to type `token` (or pass --yes)."""
    print("\n".join(summary_lines))
    if assume_yes:
        print("--yes given: proceeding without the interactive check.")
        return
    if not sys.stdin.isatty():
        raise SystemExit("Refused: no terminal to confirm on. Re-run interactively, or pass --yes.")
    answer = input(f"Type '{token}' to spend this money, anything else to stop: ").strip()
    if answer != token:
        raise SystemExit("Stopped; nothing was sent.")
