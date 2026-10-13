# Driver used for the frozen test run on 2026-09-27, copied here unchanged after the run (it was kept
# outside the repository at the time). Run from the repository root: python scripts/test_run.py (PAID).
"""Frozen test run: upload, then batches of 250 one after another, stopping on cost or failures."""
import subprocess, sys
sys.path.insert(0, ".")
from ecgcode import api, runs, spend
RUN, CHUNK, MAX_CHUNK_USD, MAX_FAIL = "runs/test_v2", 250, 20.0, 0.10
def run(*cmd):
    r = subprocess.run([sys.executable, "-m", "ecgcode", *cmd], capture_output=True, text=True)
    if r.returncode != 0:
        print("STOP:", " ".join(cmd), r.stdout[-800:], r.stderr[-800:], flush=True); sys.exit(1)
    return r.stdout
print(run("upload", "--run", RUN).strip().splitlines()[-1], flush=True)
k = 0
while True:
    todo = api.pending_requests(RUN)
    if not todo:
        break
    k += 1
    before, n_before = spend.spent_total(), runs.parse_run(RUN)["records_with_model_prediction"]
    run("submit", "--run", RUN, "--limit", str(CHUNK), "--max-usd", "70", "--yes")
    run("collect", "--run", RUN, "--poll-seconds", "60")
    s = runs.parse_run(RUN)
    cost, sent = spend.spent_total() - before, min(CHUNK, len(todo))
    fail = 1 - (s["records_with_model_prediction"] - n_before) / sent
    print(f"batch {k}: sent {sent}, answered {s['records_with_model_prediction']}/2158, cost ${cost:.2f}, "
          f"failed {fail:.0%}, project spend ${spend.spent_total():.2f}", flush=True)
    if cost > MAX_CHUNK_USD or fail > MAX_FAIL:
        print("STOP: batch cost or failure share over the limit", flush=True); sys.exit(1)
    if k > 12:
        print("STOP: more batches than expected", flush=True); sys.exit(1)
s = runs.parse_run(RUN)
print(f"DONE: answered {s['records_with_model_prediction']}/2158, prior-filled {s['records_prior_filled']}, "
      f"failures {s['failure_reasons']}, cost of all attempts ${s['usd_total_all_attempts']:.2f}", flush=True)
