"""Offline tests (no API calls): record files, build, request shape, resume logic, money stops, parse."""
import itertools
import json
import os
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest

from ecgcode import api, prompting, runs, spend
from ecgcode.config import load_config, project_path
from ecgcode.records import LEADS, record_csv

DATA_DIR = project_path(load_config()["data"]["ptbxl_dir"])
needs_data = pytest.mark.skipif(not DATA_DIR.exists(), reason="PTB-XL missing")
_IDS = itertools.count(1)


@pytest.fixture(autouse=True)
def isolated_ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("ECGREAD_LEDGER", str(tmp_path / "ledger.jsonl"))


@pytest.fixture
def run_dir(tmp_path):
    run = tmp_path / "dev3"
    runs.build_run(load_config(), run, "dev", n=3, sample_seed=1)
    (run / "files.json").write_text(json.dumps({a: f"file_{a}" for a in json.loads((run / "key.json").read_text())}))
    return run


def _usage(inp=1000, out=500):
    d = {"input_tokens": inp, "output_tokens": out, "cache_read_input_tokens": 0,
         "cache_creation_input_tokens": 0, "server_tool_use": {"code_execution_requests": 1}}
    return SimpleNamespace(to_dict=lambda: dict(d))


def _msg(stop, blocks, inp=1000, out=500):
    content = [SimpleNamespace(type=b["type"], text=b.get("text", ""), thinking=b.get("thinking", ""),
                               to_dict=(lambda b=b: dict(b))) for b in blocks]
    return SimpleNamespace(id=f"msg_{next(_IDS)}", model="claude-opus-5-5", stop_reason=stop, content=content,
                           usage=_usage(inp, out), container=SimpleNamespace(id="container_1"))


def _answer(anon):
    return json.dumps({"records": [{"record_id": anon, "key_findings": "x", "p_NORM": 0.9, "p_MI": 0.1,
                                    "p_STTC": 0.1, "p_CD": 0.1, "p_HYP": 0.05}]})


class FakeClient:
    def __init__(self, script):
        self.script, self.calls = script, []
        self.messages = SimpleNamespace(stream=self._stream)

    @contextmanager
    def _stream(self, **params):
        self.calls.append(params)
        msg = self.script(params, len(self.calls))
        if isinstance(msg, Exception):
            raise msg
        yield SimpleNamespace(get_final_message=lambda: msg)


def test_record_csv_has_all_leads_in_mv():
    sig = np.zeros((5000, 12))
    sig[100, 1] = 1.2345
    text = record_csv(sig, 500, 3)
    lines = text.splitlines()
    assert lines[0] == "t_ms," + ",".join(LEADS) and len(lines) == 5001
    assert lines[101].split(",")[0] == "200" and lines[101].split(",")[2] == "1.234"


@needs_data
def test_build_writes_files_and_placeholder_requests(run_dir):
    reqs = runs.load_requests(run_dir)
    assert len(reqs) == 3 and len(list((run_dir / "files").glob("*.csv"))) == 3
    content = reqs[0]["params"]["messages"][0]["content"]
    assert content[1] == {"type": "container_upload", "file_id": prompting.FILE_ID_PLACEHOLDER}
    assert reqs[0]["params"]["tools"][0]["type"] == "code_execution_20260521"
    filled = prompting.with_file_id(reqs[0]["params"], "file_x")
    assert filled["messages"][0]["content"][1]["file_id"] == "file_x"
    assert reqs[0]["params"]["messages"][0]["content"][1]["file_id"] == prompting.FILE_ID_PLACEHOLDER
    manifest = runs.load_manifest(run_dir)
    assert "ecg_id" not in json.dumps(reqs) and str(manifest["ecg_ids"][0]) not in content[0]["text"]


@needs_data
def test_test_fold_refused_before_freeze(tmp_path):
    with pytest.raises(SystemExit):
        runs.build_run(load_config(overrides=["model.effort=max"]), tmp_path / "t", "test", n=2)


def test_resume_after_pause_turn_sums_usage_and_reuses_container():
    def script(params, k):
        if k == 1:
            return _msg("pause_turn", [{"type": "server_tool_use", "name": "code_execution"}], 1000, 400)
        assert params["container"] == "container_1"
        assert params["messages"][-1]["role"] == "assistant"
        return _msg("end_turn", [{"type": "server_tool_use"}, {"type": "text", "text": _answer("d0001")}], 3000, 600)
    client = FakeClient(script)
    params = {"model": "m", "max_tokens": 10, "messages": [{"role": "user", "content": "x"}]}
    msgs, rec = api.call_with_resume(client, params, max_continuations=2)
    assert len(msgs) == 2 and rec["n_calls"] == 2 and rec["code_runs"] == 2
    assert rec["usage"]["input_tokens"] == 4000 and rec["usage"]["output_tokens"] == 1000
    assert rec["usage"]["server_tool_use"]["code_execution_requests"] == 2


@needs_data
def test_sync_stops_on_actual_spend(run_dir, monkeypatch):
    def script(params, k):
        anon = params["messages"][0]["content"][0]["text"].split()[1].rstrip(".")
        return _msg("end_turn", [{"type": "text", "text": _answer(anon)}], 100_000, 20_000)  # $0.80 each
    client = FakeClient(script)
    monkeypatch.setattr(api, "_client", lambda max_retries=0: client)
    cost = load_config()["cost"]
    spent = api.run_sync(run_dir, runs.load_requests(run_dir), cost, max_usd=3.0, max_continuations=1,
                         usd_worst_per_request=2.0)
    assert len(client.calls) == 2 and spent == pytest.approx(1.6)  # 0.8 + 2.0 > 3.0 after the second
    assert spend.spent_total() == pytest.approx(1.6)
    summary = runs.parse_run(run_dir)
    assert summary["records_with_model_prediction"] == 2 and summary["failure_reasons"] == {"not_sent": 1}


@needs_data
def test_sync_client_error_stops_without_recording(run_dir, monkeypatch):
    import anthropic
    import httpx

    def script(params, k):
        resp = httpx.Response(400, request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"))
        return anthropic.BadRequestError("bad shape", response=resp, body=None)
    monkeypatch.setattr(api, "_client", lambda max_retries=0: FakeClient(script))
    with pytest.raises(SystemExit):
        api.run_sync(run_dir, runs.load_requests(run_dir), load_config()["cost"], 5.0, 1, 2.0)
    assert runs.load_results(run_dir) == []


def test_validate_answer_accepts_bare_object_and_rejects_nan():
    ok = {"type": "succeeded", "stop_reason": "end_turn"}
    bare = json.dumps({"record_id": "d0001", "key_findings": "", "p_NORM": 0.2, "p_MI": 0.1, "p_STTC": 0.3,
                       "p_CD": 0.4, "p_HYP": 0.1})
    preds, fails, _ = runs.validate_answer({**ok, "text": bare}, ["d0001"])
    assert preds["d0001"]["p_CD"] == 0.4 and not fails
    preds, fails, _ = runs.validate_answer({**ok, "text": _answer("d0001").replace("0.9", "NaN")}, ["d0001"])
    assert fails == {"d0001": "unparseable_json"} or fails == {"d0001": "invalid_probability"}
    assert runs.validate_answer({**ok, "stop_reason": "pause_turn", "text": ""}, ["d0001"])[1] == {
        "d0001": "stop_pause_turn"}


class FakeBatches:
    def __init__(self):
        self.created, self.results_by_batch = [], {}

    def create(self, requests):
        self.created.append(requests)
        return SimpleNamespace(id=f"msgbatch_{len(self.created)}")

    def retrieve(self, batch_id):
        return SimpleNamespace(id=batch_id, processing_status="ended")

    def results(self, batch_id):
        return self.results_by_batch.get(batch_id, [])


@needs_data
def test_batch_submit_collect_resumes_paused_result(run_dir, monkeypatch):
    reqs = runs.load_requests(run_dir)
    batches = FakeBatches()

    def stream_script(params, k):  # sync continuation of the paused batch result
        assert params["container"] == "container_1" and params["messages"][-1]["role"] == "assistant"
        anon = params["messages"][0]["content"][0]["text"].split()[1].rstrip(".")
        return _msg("end_turn", [{"type": "text", "text": _answer(anon)}], 2000, 300)
    client = FakeClient(stream_script)
    client.messages.batches = batches
    monkeypatch.setattr(api, "_client", lambda max_retries=0: client)
    cost = load_config()["cost"]
    api.submit_batches(run_dir, reqs, usd_worst_per_request=0.5, cost=cost)
    sent = batches.created[0]
    assert len(sent) == 3 and all(r["params"]["messages"][0]["content"][1]["file_id"].startswith("file_") for r in sent)
    with pytest.raises(SystemExit):
        api.guard_inflight(run_dir)                                  # no second submit while in flight
    assert api.pending_requests(run_dir) == [] and api.committed_usd([run_dir]) == pytest.approx(1.5)
    items = []
    for k, r in enumerate(reqs):
        anon = r["record_ids"][0]
        msg = (_msg("pause_turn", [{"type": "server_tool_use"}], 1000, 200) if k == 0
               else _msg("end_turn", [{"type": "text", "text": _answer(anon)}], 1000, 200))
        items.append(SimpleNamespace(custom_id=r["custom_id"], result=SimpleNamespace(type="succeeded", message=msg)))
    batches.results_by_batch["msgbatch_1"] = items
    api.collect_batches(run_dir, cost, max_continuations=1)
    results = runs.load_results(run_dir)
    paused = [r for r in results if r["custom_id"] == reqs[0]["custom_id"]][0]
    assert paused["n_calls"] == 2 and paused["resumed_sync_calls"] == 1 and paused["stop_reason"] == "end_turn"
    batch_part = spend.usage_cost_usd({"input_tokens": 1000, "output_tokens": 200}, cost, "batch")
    sync_part = spend.usage_cost_usd({"input_tokens": 2000, "output_tokens": 300}, cost, "sync")
    assert paused["usd"] == pytest.approx(batch_part + sync_part, abs=1e-6)
    assert api.committed_usd([run_dir]) == 0.0 and spend.spent_total() == pytest.approx(3 * batch_part + sync_part)
    assert runs.parse_run(run_dir)["records_with_model_prediction"] == 3


@needs_data
def test_examples_come_from_training_folds_and_are_attached(tmp_path):
    from ecgcode.data import TRAIN_FOLDS, load_metadata
    cfg = load_config(overrides=["examples.n_per_class=1", "prompt.system=prompts/reader_code_v1_examples.md"])
    run = tmp_path / "ex"
    runs.build_run(cfg, run, "dev", n=2, sample_seed=1)
    ids = json.loads((run / "examples" / "example_ids.json").read_text())
    meta = load_metadata(DATA_DIR)
    assert len(ids) == 5 and all(meta.loc[i, "strat_fold"] in TRAIN_FOLDS for i in ids.values())
    labels = (run / "examples" / "example_labels.csv").read_text()
    assert "superclasses" in labels and str(list(ids.values())[0]) not in labels  # no PTB-XL ids exposed
    req = runs.load_requests(run)[0]
    blocks = req["params"]["messages"][0]["content"]
    assert sum(b["type"] == "container_upload" for b in blocks) == 1 + 5 + 1
    files = {a: f"f_{a}" for a in json.loads((run / "key.json").read_text())}
    with pytest.raises(SystemExit):                       # examples not uploaded yet
        prompting.with_file_id(req["params"], "f_rec", files)
    files.update({f"examples/{p.name}": f"f_{p.name}" for p in (run / "examples").glob("*.csv")})
    filled = prompting.with_file_id(req["params"], "f_rec", files)
    assert not any("{" in b.get("file_id", "") for b in filled["messages"][0]["content"])


def test_statement_level_answers_are_aggregated_by_class():
    from ecgcode.statements import STATEMENT_CLASSES
    stm = {c: 0.0 for c in STATEMENT_CLASSES}
    stm.update({"LVH": 0.7, "LAO/LAE": 0.4, "IMI": 0.3, "NORM": 0.1})
    entry = {"record_id": "d0001", "key_findings": "", "statements": stm, "p_NORM": 0.2, "p_MI": 0.2,
             "p_STTC": 0.2, "p_CD": 0.2, "p_HYP": 0.5}
    res = {"type": "succeeded", "stop_reason": "end_turn", "text": json.dumps({"records": [entry]})}
    preds, fails, _ = runs.validate_answer(res, ["d0001"], "max")
    p = preds["d0001"]
    assert (p["p_HYP"], p["p_MI"], p["p_NORM"], p["p_CD"]) == (0.7, 0.3, 0.1, 0.0) and p["direct_p_HYP"] == 0.5
    preds, _, _ = runs.validate_answer(res, ["d0001"], "noisy_or")
    assert preds["d0001"]["p_HYP"] == pytest.approx(1 - 0.3 * 0.6)
    bad = {**entry, "statements": {**stm, "LVH": "high"}}
    _, fails, _ = runs.validate_answer({**res, "text": json.dumps({"records": [bad]})}, ["d0001"], "max")
    assert fails == {"d0001": "invalid_statements"}
