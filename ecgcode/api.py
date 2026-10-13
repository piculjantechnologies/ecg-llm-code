"""Anthropic API calls (the only module that talks to the API).

Free: upload/delete record files (Files API), preflight (models.retrieve, count_tokens).
Paid: submit (--sync for now), behind the budget gate in cli.py.

Code-tool requests run a server-side loop: the model writes code, the sandbox runs it and the
output is fed back, so one request can use many times its first input and can end with
stop_reason "pause_turn" (the loop limit), after which it is resumed in the same container.
Every call of a request is billed and summed into one result line. Safety rules:
- sync requests run one at a time, and sending stops once the actual cost of this submission
  reaches --max-usd minus the assumed worst cost of one request;
- any 4xx other than 408/409/429 (credentials, billing, request shape) stops the run and is not
  recorded as an attempt; 429/5xx are retried up to 4 times before anything was generated;
- a dropped connection or unexpected error is recorded as a possibly billed attempt and stops the run.
"""
import datetime as dt
import json
import time
from pathlib import Path

from .prompting import with_file_id
from .runs import append_results, load_requests, load_results, request_status
from .spend import record_possible_spend, record_spend, usage_cost_usd

RETRYABLE_STATUS = {408, 409, 429}


def _client(max_retries=0):
    import anthropic  # lazy: offline commands work without the SDK configured

    return anthropic.Anthropic(max_retries=max_retries)


def _now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


# --- record files (Files API, free) --------------------------------------------------------

def _file_map(run_dir):
    path = Path(run_dir) / "files.json"
    return json.loads(path.read_text()) if path.exists() else {}


def upload_files(run_dir, anons=None):
    """Upload each record's CSV once; files.json maps anon -> file id (saved after every upload)."""
    run_dir = Path(run_dir)
    files = _file_map(run_dir)
    todo = [(p.stem, p) for p in sorted((run_dir / "files").glob("*.csv"))
            if p.stem not in files and (anons is None or p.stem in anons)]
    todo += [(f"examples/{p.name}", p) for p in sorted((run_dir / "examples").glob("*.csv"))
             if f"examples/{p.name}" not in files]
    client = _client(max_retries=2)
    for k, (key, path) in enumerate(todo, 1):
        meta = client.beta.files.upload(file=(path.name, path.read_bytes(), "text/csv"))
        files[key] = meta.id
        tmp = run_dir / "files.json.tmp"
        tmp.write_text(json.dumps(files, indent=1))
        tmp.replace(run_dir / "files.json")
        if k % 50 == 0:
            print(f"uploaded {k}/{len(todo)}")
    print(f"{len(files)} record files uploaded ({len(todo)} new)")
    return files


def delete_files(run_dir):
    """Delete this run's uploaded record files (free); the local copies stay in files/."""
    run_dir = Path(run_dir)
    files = _file_map(run_dir)
    client = _client(max_retries=2)
    for anon, file_id in list(files.items()):
        client.beta.files.delete(file_id)
        del files[anon]
        (run_dir / "files.json").write_text(json.dumps(files, indent=1))
    print("uploaded record files deleted")


# --- what may be sent ------------------------------------------------------------------------

def pending_requests(run_dir, limit=None, max_attempts=2):
    status = request_status(run_dir)
    in_flight, _ = inflight(run_dir)
    todo = [r for r in load_requests(run_dir)
            if not status[r["custom_id"]]["complete"] and status[r["custom_id"]]["attempts"] < max_attempts
            and status[r["custom_id"]]["last_failure"] != "stop_max_tokens" and r["custom_id"] not in in_flight]
    return todo[:limit] if limit else todo


def calibration(model_id, prompt_sha, runs_root):
    """Mean cost per answered request from earlier results with the same model and prompt."""
    costs = []
    for manifest_path in Path(runs_root).glob("*/manifest.json"):
        try:
            m = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            continue
        if m["config"]["model"]["id"] != model_id or m.get("prompt_sha256") != prompt_sha:
            continue
        costs += [float(r["usd"]) for r in load_results(manifest_path.parent)
                  if r.get("type") == "succeeded" and r.get("usd") is not None]
    if len(costs) < 5:
        return None
    ordered = sorted(costs)
    return {"n": len(ordered), "mean_usd": sum(ordered) / len(ordered), "max_usd": ordered[-1]}


def plan_cost(run_dir, todo, cost, mode="sync", runs_root=None):
    from .runs import load_manifest

    manifest = load_manifest(run_dir)
    discount = cost["batch_discount"] if mode == "batch" else 1.0
    cal = calibration(manifest["config"]["model"]["id"], manifest["prompt_sha256"],
                      runs_root or Path(run_dir).parent)
    n = len(todo)
    worst_per_request = float(cost["assumed_worst_usd_per_request"]) * discount
    expected_per_request = (cal["mean_usd"] * (discount if mode == "batch" else 1.0) if cal
                            else worst_per_request / 4)
    return {"requests": n, "mode": mode, "calibration": cal,
            "usd_expected": round(n * expected_per_request, 2), "usd_worst_case": round(n * worst_per_request, 2),
            "usd_worst_per_request": worst_per_request,
            "note": "worst case is an assumption for code-tool loops until measured; sync sending also "
                    "stops on actual spend"}


# --- one request with resumptions ------------------------------------------------------------

def _block_dict(block):
    return block.to_dict() if hasattr(block, "to_dict") else dict(block)


def _add_usage(total, usage):
    for k, v in usage.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            total[k] = total.get(k, 0) + v
        elif isinstance(v, dict):
            sub = total.setdefault(k, {})
            for k2, v2 in v.items():
                if isinstance(v2, (int, float)) and not isinstance(v2, bool):
                    sub[k2] = sub.get(k2, 0) + v2
    return total


def call_with_resume(client, params, max_continuations, first=None):
    """Run one request, resuming after pause_turn in the same container. Returns (messages, record).
    `first` is an already received message (e.g. a batch result) to continue from."""
    messages, assistant_blocks = [], []
    p = dict(params)
    if first is not None:
        messages.append(first)
        assistant_blocks += [_block_dict(b) for b in first.content]
        if first.stop_reason != "pause_turn":
            return messages, record_from_messages(messages, assistant_blocks)
        p["messages"] = params["messages"] + [{"role": "assistant", "content": assistant_blocks}]
        container = getattr(first, "container", None)
        if container is not None and getattr(container, "id", None):
            p["container"] = container.id
    n_new_calls = max_continuations + 1 if first is None else max_continuations
    for _ in range(n_new_calls):
        with client.messages.stream(**p) as stream:
            msg = stream.get_final_message()
        messages.append(msg)
        assistant_blocks += [_block_dict(b) for b in msg.content]
        if msg.stop_reason != "pause_turn":
            break
        p = dict(params)
        p["messages"] = params["messages"] + [{"role": "assistant", "content": assistant_blocks}]
        container = getattr(msg, "container", None)
        if container is not None and getattr(container, "id", None):
            p["container"] = container.id
    return messages, record_from_messages(messages, assistant_blocks)


def record_from_messages(messages, assistant_blocks=None):
    """One result line from the API message(s) of one request (summed usage, final text, code runs)."""
    if assistant_blocks is None:
        assistant_blocks = [_block_dict(b) for m in messages for b in m.content]
    usage = {}
    for m in messages:
        _add_usage(usage, m.usage.to_dict())
    last = messages[-1]
    text = "".join(b.text for b in last.content if b.type == "text")
    thinking = "\n".join(getattr(b, "thinking", "") or "" for m in messages for b in m.content
                         if b.type == "thinking")
    code_runs = sum(1 for b in assistant_blocks if b.get("type") == "server_tool_use")
    container = getattr(last, "container", None)
    record = {"type": "succeeded", "message_id": messages[0].id, "message_ids": [m.id for m in messages],
              "model": last.model, "stop_reason": last.stop_reason, "n_calls": len(messages),
              "text": text, "thinking_summary": thinking, "usage": usage, "code_runs": code_runs,
              "container_id": getattr(container, "id", None), "content": assistant_blocks}
    return record


# --- batches ---------------------------------------------------------------------------------

MAX_BATCH_REQUESTS = 10_000


def _batch_log(run_dir):
    path = Path(run_dir) / "batches.json"
    return json.loads(path.read_text()) if path.exists() else []


def _write_batch_log(run_dir, log):
    path = Path(run_dir) / "batches.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(log, indent=2))
    tmp.replace(path)


def inflight(run_dir):
    """(custom_ids in uncollected batches, uncollected entries without a known batch id)."""
    ids, unresolved = set(), []
    for entry in _batch_log(run_dir):
        if entry.get("collected"):
            continue
        if not entry.get("id"):
            unresolved.append(entry)
        ids |= set(entry.get("custom_ids") or [])
    return ids, unresolved


def guard_inflight(run_dir):
    ids, unresolved = inflight(run_dir)
    if unresolved:
        raise SystemExit(f"{run_dir}: a batch creation has an unknown outcome (batches.json); resolve it with "
                         f"`python -m ecgcode reconcile --run {run_dir}` before sending anything.")
    if ids:
        raise SystemExit(f"{run_dir}: {len(ids)} requests are in batches that have not been collected; "
                         f"run `python -m ecgcode collect --run {run_dir}` first.")


def committed_usd(run_dirs):
    """Assumed worst cost of batches submitted but not yet collected."""
    return sum(float(e.get("usd_worst_case") or 0.0) for d in run_dirs for e in _batch_log(d)
               if not e.get("collected"))


def all_run_dirs(extra=()):
    from .config import project_path

    dirs = {p.parent.resolve() for p in project_path("runs").glob("*/manifest.json")}
    return sorted(dirs | {Path(d).resolve() for d in extra})


def submit_batches(run_dir, todo, usd_worst_per_request, cost):
    """Create batches for `todo` (budget-checked by the caller). Creation is never retried blindly."""
    run_dir = Path(run_dir)
    files = _file_map(run_dir)
    missing = [a for r in todo for a in r["record_ids"] if a not in files]
    if missing:
        raise SystemExit(f"{len(missing)} record files are not uploaded yet; run `upload` first (free).")
    client = _client(max_retries=0)
    for start in range(0, len(todo), MAX_BATCH_REQUESTS):
        chunk = todo[start:start + MAX_BATCH_REQUESTS]
        log = _batch_log(run_dir)
        entry = {"id": None, "state": "creating", "started_utc": _now(), "n_requests": len(chunk),
                 "custom_ids": [r["custom_id"] for r in chunk], "collected": False,
                 "usd_worst_case": round(usd_worst_per_request * len(chunk), 2), "cost": cost}
        log.append(entry)
        _write_batch_log(run_dir, log)  # before the paid call
        try:
            batch = client.messages.batches.create(requests=[
                {"custom_id": r["custom_id"], "params": with_file_id(r["params"], files[r["record_ids"][0]], files)}
                for r in chunk])
        except Exception as exc:  # the batch may or may not exist on the server
            entry.update({"state": "create_failed_unknown", "error": f"{type(exc).__name__}: {exc}"})
            _write_batch_log(run_dir, log)
            raise SystemExit(f"Batch creation failed ({type(exc).__name__}: {exc}). It may still exist; "
                             f"run `python -m ecgcode reconcile --run {run_dir}` before sending anything.")
        entry.update({"id": batch.id, "state": "submitted", "submitted_utc": _now()})
        _write_batch_log(run_dir, log)
        print(f"Submitted batch {batch.id} with {len(chunk)} requests")


def reconcile(run_dir, attach=None, drop_unknown=False, limit=20):
    run_dir = Path(run_dir)
    log = _batch_log(run_dir)
    unresolved = [e for e in log if not e.get("id") and not e.get("collected")]
    if not unresolved:
        print("Nothing to reconcile.")
        return
    if attach:
        b = _client(max_retries=2).messages.batches.retrieve(attach)
        c = b.request_counts
        total = c.processing + c.succeeded + c.errored + c.canceled + c.expired
        if total != unresolved[0]["n_requests"]:
            raise SystemExit(f"Refused: batch {attach} has {total} requests, the entry {unresolved[0]['n_requests']}.")
        unresolved[0].update({"id": attach, "state": "submitted", "attached_utc": _now()})
        _write_batch_log(run_dir, log)
        print(f"Attached {attach} (created {b.created_at}); now run collect.")
        return
    if drop_unknown:
        for e in unresolved:
            e.update({"state": "not_created", "collected": True, "custom_ids": []})
        _write_batch_log(run_dir, log)
        print("Marked as not created; their requests are pending again.")
        return
    print(f"Unresolved entry: started {unresolved[0]['started_utc']}, {unresolved[0]['n_requests']} requests.")
    for b in _client(max_retries=2).messages.batches.list(limit=limit).data:
        c = b.request_counts
        print(f"  {b.id}  created {b.created_at}  status {b.processing_status}  "
              f"requests {c.processing + c.succeeded + c.errored + c.canceled + c.expired}")
    print("Then: reconcile --attach BATCH_ID, or --drop-unknown if none matches.")


def cancel_batches(run_dir):
    client = _client(max_retries=2)
    for entry in _batch_log(run_dir):
        if entry.get("id") and not entry.get("collected"):
            print(f"{entry['id']}: {client.messages.batches.cancel(entry['id']).processing_status}")


def wait_for_batches(run_dir, poll_seconds=60):
    client = _client(max_retries=3)
    for entry in _batch_log(run_dir):
        if entry.get("collected") or not entry.get("id"):
            continue
        while True:
            b = client.messages.batches.retrieve(entry["id"])
            if b.processing_status == "ended":
                break
            c = b.request_counts
            print(f"{entry['id']}: {b.processing_status} (processing {c.processing}, succeeded {c.succeeded}, "
                  f"errored {c.errored})")
            time.sleep(poll_seconds)


def collect_batches(run_dir, cost_fallback, max_continuations):
    """Download ended batches. Results that ended with pause_turn are resumed with sync calls (full
    price); the batch part is billed at batch prices. Everything is recorded before the entry is
    marked collected."""
    run_dir = Path(run_dir)
    log = _batch_log(run_dir)
    requests = {r["custom_id"]: r for r in load_requests(run_dir)}
    files = _file_map(run_dir)
    client = _client(max_retries=3)
    for entry in log:
        if entry.get("collected") or not entry.get("id"):
            continue
        if client.messages.batches.retrieve(entry["id"]).processing_status != "ended":
            print(f"{entry['id']}: not ended yet; run collect again later")
            continue
        prices = entry.get("cost") or cost_fallback
        expected = set(entry.get("custom_ids") or [])
        items = list(client.messages.batches.results(entry["id"]))
        foreign = [i.custom_id for i in items if i.custom_id not in expected]
        if foreign:
            raise SystemExit(f"{entry['id']}: {len(foreign)} results were not sent by this entry; nothing recorded.")
        records = []
        for item in items:
            record = {"custom_id": item.custom_id, "mode": "batch", "batch_id": entry["id"],
                      "collected_utc": _now(), "type": item.result.type}
            if item.result.type == "succeeded":
                msg = item.result.message
                usd = usage_cost_usd(msg.usage.to_dict(), prices, "batch")
                if msg.stop_reason == "pause_turn":
                    req = requests[item.custom_id]
                    params = with_file_id(req["params"], files[req["record_ids"][0]], files)
                    messages, rec = call_with_resume(client, params, max_continuations, first=msg)
                    usd += sum(usage_cost_usd(m.usage.to_dict(), prices, "sync") for m in messages[1:])
                    record["resumed_sync_calls"] = len(messages) - 1
                else:
                    rec = record_from_messages([msg])
                record.update(rec)
                record["usd"] = round(usd, 6)
            elif item.result.type == "errored":
                record["error"] = item.result.error.to_dict()
            records.append(record)
        append_results(run_dir, records)
        record_spend(run_dir, records, prices)
        entry.update({"collected": True, "collected_utc": _now(), "n_results": len(records)})
        _write_batch_log(run_dir, log)
        print(f"{entry['id']}: collected {len(records)} results")


# --- synchronous submission ------------------------------------------------------------------

def run_sync(run_dir, todo, cost, max_usd, max_continuations, usd_worst_per_request):
    """Send requests one at a time; stop before the actual cost of this submission could pass max_usd."""
    import anthropic
    import httpx

    run_dir = Path(run_dir)
    files = _file_map(run_dir)
    missing = [a for r in todo for a in r["record_ids"] if a not in files]
    if missing:
        raise SystemExit(f"{len(missing)} record files are not uploaded yet; run `upload` first (free).")
    client = _client(max_retries=0)
    spent, sent = 0.0, 0
    # fail early if an example file is missing
    for req in todo[:1]:
        with_file_id(req["params"], files[req["record_ids"][0]], files)
    for req in todo:
        if spent + usd_worst_per_request > max_usd:
            print(f"Stopping: ${spent:.2f} spent in this submission; one more request could pass --max-usd "
                  f"{max_usd:.2f}. {len(todo) - sent} request(s) not sent.")
            break
        params = with_file_id(req["params"], files[req["record_ids"][0]], files)
        record = {"custom_id": req["custom_id"], "mode": "sync", "collected_utc": _now()}
        fatal = None
        for attempt in range(4):
            try:
                _, result = call_with_resume(client, params, max_continuations)
                record.update(result)
                record["usd"] = round(usage_cost_usd(result["usage"], cost, "sync"), 6)
                break
            except anthropic.APIStatusError as e:
                if e.status_code < 400:  # error event inside a started stream: possibly billed
                    record.update({"type": "errored", "error": {"status": e.status_code,
                                                                "message": f"{e.message} (possibly billed)"}})
                    fatal = f"error during the stream: {e.message}"
                    break
                if not (e.status_code >= 500 or e.status_code in RETRYABLE_STATUS):
                    raise SystemExit(f"Stopped at a client error (not billed, not an attempt): "
                                     f"{type(e).__name__} ({e.status_code}): {e.message}")
                if attempt == 3:
                    raise SystemExit(f"Stopped: {type(e).__name__} ({e.status_code}) four times; try later.")
                time.sleep(min(60, 5 * 2 ** attempt))
            except (anthropic.APITimeoutError, anthropic.APIConnectionError, httpx.TransportError) as e:
                record.update({"type": "errored", "error": {"status": None,
                                                            "message": f"{type(e).__name__}: {e} (possibly billed)"}})
                fatal = f"connection problem ({type(e).__name__})"
                break
        append_results(run_dir, [record])
        record_spend(run_dir, [record], cost)
        if record.get("type") != "succeeded":
            record_possible_spend(run_dir, record["custom_id"], usd_worst_per_request,
                                  "sync code-tool request interrupted: counted at its assumed worst case")
            spent += usd_worst_per_request
        else:
            spent += record["usd"]
        sent += 1
        print(f"{record['custom_id']}: {record.get('stop_reason', record.get('type'))}, "
              f"{record.get('code_runs', 0)} code runs, {record.get('n_calls', 0)} call(s), "
              f"${record.get('usd', 0):.3f} (submission total ${spent:.2f})")
        if fatal:
            raise SystemExit(f"Stopped after a possibly billed failure: {fatal}")
    return spent


# --- free checks ----------------------------------------------------------------------------

def preflight(run_dir):
    """Credentials and model availability; request shape via count_tokens (tools included)."""
    run_dir = Path(run_dir)
    req = load_requests(run_dir)[0]
    client = _client(max_retries=2)
    info = client.models.retrieve(req["params"]["model"])
    print(f"credentials OK; model {info.id} available")
    files = _file_map(run_dir)
    params = req["params"]
    if req["record_ids"][0] in files:
        params = with_file_id(params, files[req["record_ids"][0]], files)
    keys = ("model", "system", "messages", "thinking", "output_config", "tools")
    try:
        n = client.messages.count_tokens(**{k: params[k] for k in keys if k in params}).input_tokens
        print(f"count_tokens accepted the request shape: {n} input tokens before any code runs")
    except Exception as e:  # count_tokens may not support every server-tool/file combination
        print(f"count_tokens did not accept the full request ({type(e).__name__}: {e}); "
              f"the first sync request will show whether the API does")
