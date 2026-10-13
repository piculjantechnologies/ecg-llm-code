"""Pre-specified results of the test run and the development results.

Run from the repository root:  python analysis/results.py
Writes paper/generated/results.tex and analysis/results.json. No API calls.

Pre-specified: macro AUROC with 95% CI on all 2,158 test records, per-class
AUROC, macro AUPRC, the paired difference against the age-and-sex control (analysis/agesex_control.py),
code runs, tokens and cost. The primary endpoint and the paired difference are computed with the frozen
ecgcode.evaluate functions. Per-class confidence intervals were not pre-specified; they use the same
bootstrap resamples. Development results use validation-fold runs made before the freeze.
"""
import datetime as dt
import json

import numpy as np

from common import (AGESEX_DEV, AGESEX_TEST, CFG, GEN, LABELS, OUT_JSON, ROOT, RUNS, TEST,
                    TEST_FOLD, TRAIN_FOLDS, VAL_FOLD, benchmark_split, boots_for, f2, f3, long_date, meta, paired,
                    preds, results_lines, s3, summary, thousands, write_json, write_macros)
from ecgcode.evaluate import evaluate_run, paired_difference
from ecgcode.spend import usage_cost_usd

DEV_RUNS = ["pilot10", "dev500", "dev100_measure", "dev400_measure", "dev100_examples",
            "dev100_measure_examples", "dev100_statements", "dev200b_v2", "dev200b_v5"]


def run_usd(run):
    return json.loads((RUNS / run / "usage.json").read_text())["usd_total_all_attempts"]


def run_window(run_dir):
    """First submission and last collection of a run, from its batch log. A request sent on its own (sync)
    counts with the time it was sent, the only time its result records."""
    starts, ends = [], []
    log = run_dir / "batches.json"
    for b in json.loads(log.read_text()) if log.exists() else []:
        starts.append(dt.datetime.fromisoformat(b["started_utc"]))
        ends.append(dt.datetime.fromisoformat(b["collected_utc"]))
    for r in results_lines(run_dir):
        if r.get("mode") == "sync":
            starts.append(dt.datetime.fromisoformat(r["collected_utc"]))
            ends.append(dt.datetime.fromisoformat(r["collected_utc"]))
    return min(starts), max(ends)


def half_width(ci):
    return (ci[1] - ci[0]) / 2


def usage_stats(run_dir, cost):
    lines = [r for r in results_lines(run_dir) if r.get("type") == "succeeded"]
    u = lambda k: np.array([r["usage"].get(k, 0) for r in lines], float)  # noqa: E731
    thinking = np.array([r["usage"].get("output_tokens_details", {}).get("thinking_tokens", 0) for r in lines], float)
    code_runs = np.array([r.get("code_runs", 0) for r in lines], float)
    usd = np.array([r["usd"] for r in lines], float)
    # the part of each result's cost that paid for writing the prompt cache, priced as ecgcode.spend prices it
    writes = [{"cache_creation_input_tokens": r["usage"].get("cache_creation_input_tokens", 0)} for r in lines]
    cache_write_usd = np.array([usage_cost_usd(w, cost, r["mode"]) for w, r in zip(writes, lines)])
    calls = np.array([r.get("n_calls", 1) for r in lines], float)
    stops = {}
    for r in lines:
        stops[r["stop_reason"]] = stops.get(r["stop_reason"], 0) + 1
    input_total = u("input_tokens") + u("cache_read_input_tokens") + u("cache_creation_input_tokens")
    return {
        "n": len(lines),
        "usd_mean": float(usd.mean()), "usd_median": float(np.median(usd)), "usd_max": float(usd.max()),
        "usd_p95": float(np.percentile(usd, 95)),
        "code_runs_mean": float(code_runs.mean()), "code_runs_median": float(np.median(code_runs)),
        "code_runs_max": int(code_runs.max()), "code_runs_min": int(code_runs.min()),
        "output_tokens_mean": float(u("output_tokens").mean()),
        "output_tokens_median": float(np.median(u("output_tokens"))),
        "thinking_tokens_mean": float(thinking.mean()),
        "input_tokens_mean": float(input_total.mean()),
        "cache_read_share": float(u("cache_read_input_tokens").sum() / input_total.sum()),
        "cache_write_cost_share": float(cache_write_usd.sum() / usd.sum()),
        "api_calls_mean": float(calls.mean()), "resumed": int((calls > 1).sum()),
        "stop_reasons": stops,
    }


def main():
    m = meta()
    cfg = json.loads((TEST / "manifest.json").read_text())["config"]
    test_manifest = json.loads((TEST / "manifest.json").read_text())
    usage = json.loads((TEST / "usage.json").read_text())
    ids = sorted(preds(TEST).index)
    boots = boots_for(ids)

    # --- pre-specified: primary and secondary endpoints (frozen evaluation code) -----------------
    primary = evaluate_run(TEST, cfg)
    ext = summary(preds(TEST), ids, boots)
    assert abs(ext["macro_auroc"] - primary["macro_auroc"]) < 1e-12
    vs_agesex = paired_difference(TEST, AGESEX_TEST, cfg)
    vs_agesex["a"], vs_agesex["b"] = (p.relative_to(ROOT).as_posix() for p in (TEST, AGESEX_TEST))
    agesex_sum = summary(preds(AGESEX_TEST), ids, boots)
    vs_agesex_pc = paired(preds(TEST), preds(AGESEX_TEST), ids, boots)["per_class"]
    use = usage_stats(TEST, cfg["cost"])

    # --- development (validation fold, before the freeze) -------------------------------------
    # The paper reports the frozen prompt (v2) and the variants compared against it; earlier prompt versions are
    # not reported. Their runs still count towards the records, runs and cost of development.
    v2 = preds(RUNS / "dev100_measure", RUNS / "dev400_measure")
    dev500 = sorted(v2.index)
    assert len(dev500) == 500
    ids100 = sorted(preds(RUNS / "dev100_measure").index)
    ids200 = sorted(preds(RUNS / "dev200b_v2").index)
    b500, b100, b200 = boots_for(dev500), boots_for(ids100), boots_for(ids200)
    dev = {
        "v2_500": summary(v2, dev500, b500),
        "both_vs_v2_100": paired(preds(RUNS / "dev100_measure_examples"), v2, ids100, b100),
        "both_100": summary(preds(RUNS / "dev100_measure_examples"), ids100, b100),
        "statements_vs_v2_100": paired(preds(RUNS / "dev100_statements"), v2, ids100, b100),
        "statements_100": summary(preds(RUNS / "dev100_statements"), ids100, b100),
        "v2_100": summary(v2, ids100, b100),
        "v5_vs_v2_200": paired(preds(RUNS / "dev200b_v5"), preds(RUNS / "dev200b_v2"), ids200, b200),
        "v2_200": summary(preds(RUNS / "dev200b_v2"), ids200, b200),
        "v5_200": summary(preds(RUNS / "dev200b_v5"), ids200, b200),
        # context, not development decisions: the age-and-sex control on the same validation records
        "agesex_500": summary(preds(AGESEX_DEV), dev500, b500),
        "agesex_200": summary(preds(AGESEX_DEV), ids200, b200),
    }
    all_v2 = preds(RUNS / "dev100_measure", RUNS / "dev400_measure", RUNS / "dev200b_v2")
    ids700 = sorted(all_v2.index)
    dev["v2_700"] = summary(all_v2, ids700)
    dev["agesex_700"] = summary(preds(AGESEX_DEV), ids700)
    dev_usd = {r: run_usd(r) for r in DEV_RUNS}
    # records used in development: the 500-record sample, the 200 new records and the pilot records outside both
    dev_ids = set().union(*[set(preds(RUNS / r).index) for r in DEV_RUNS])
    pilot = set(preds(RUNS / "pilot10").index)
    assert dev_ids == set(dev500) | set(ids200) | pilot and not pilot & set(ids200)
    dev_windows = [run_window(RUNS / r) for r in DEV_RUNS]
    dev_first, dev_last = min(w[0] for w in dev_windows), max(w[1] for w in dev_windows)

    # --- provenance ------------------------------------------------------------------------------
    batches = json.loads((TEST / "batches.json").read_text())
    first_submit, last_collect = run_window(TEST)
    assert first_submit.date() == last_collect.date()  # one date in the text (TestRunDate)

    n_train = len(benchmark_split(m, TRAIN_FOLDS))
    n_val = len(benchmark_split(m, VAL_FOLD))
    mac = {
        "NTest": thousands(len(ids)), "NTrain": thousands(n_train), "NVal": thousands(n_val),
        "NRecordsTotal": thousands(len(m)), "NPatients": thousands(m["patient_id"].nunique()),
        "NAnswered": thousands(usage["records_with_model_prediction"]),
        "NPriorFilled": thousands(usage["records_prior_filled"]),
        "NFailures": thousands(sum(usage["failure_reasons"].values())),
        "NClipped": thousands(usage["probabilities_clipped"]),
        "NResumed": thousands(use["resumed"]),
        "PromptSHA": "\\texttt{" + test_manifest["prompt_sha256"][:12] + "}",
        "TestRunDate": first_submit.date().isoformat(), "TestRunDateLong": long_date(first_submit),
        "TestFirstSubmitUTC": first_submit.strftime("%H:%M"),
        "TestLastCollectUTC": last_collect.strftime("%H:%M"),
        "TestHours": f"{(last_collect - first_submit).total_seconds() / 3600:.1f}",
        "NBatches": str(len(batches)),
        "DevFirstSubmitUTC": dev_first.strftime("%H:%M"), "DevFirstSubmitDateLong": long_date(dev_first),
        "DevLastCollectUTC": dev_last.strftime("%H:%M"), "DevLastCollectDateLong": long_date(dev_last),
        "NBoot": thousands(CFG["eval"]["n_bootstrap"]),
        "ModelID": "\\texttt{" + cfg["model"]["id"] + "}",
        "Effort": "\\texttt{" + cfg["model"]["effort"] + "}",
        "MaxTokens": thousands(cfg["model"]["max_tokens"]),
        "CodeTool": "\\texttt{" + cfg["model"]["code_tool"].replace("_", "\\_") + "}",
        # primary
        "MainAUROC": f3(primary["macro_auroc"]),
        "MainAUROCLow": f3(primary["macro_auroc_ci95"][0]), "MainAUROCHigh": f3(primary["macro_auroc_ci95"][1]),
        "MainAUPRC": f3(primary["macro_auprc"]),
        # comparator
        "AgeSexAUROC": f3(vs_agesex["macro_auroc_b"]),
        "AgeSexAUROCLow": f3(agesex_sum["macro_auroc_ci95"][0]),
        "AgeSexAUROCHigh": f3(agesex_sum["macro_auroc_ci95"][1]),
        "CodeVsAgeSex": s3(vs_agesex["diff"]),
        "CodeVsAgeSexLow": s3(vs_agesex["diff_ci95"][0]), "CodeVsAgeSexHigh": s3(vs_agesex["diff_ci95"][1]),
        # cost and tool use
        "CostTotal": f2(usage["usd_total_all_attempts"]),
        "CostPerRecord": f"{usage['usd_total_all_attempts'] / len(ids):.3f}",
        "CostPerRecordMedian": f"{use['usd_median']:.3f}", "CostPerRecordMax": f"{use['usd_max']:.3f}",
        "CostPerRecordPNinetyFive": f"{use['usd_p95']:.3f}",
        "CostPerThousand": f"{1000 * usage['usd_total_all_attempts'] / len(ids):.0f}",
        "CodeRunsMean": f"{use['code_runs_mean']:.1f}", "CodeRunsMedian": f"{use['code_runs_median']:.0f}",
        "CodeRunsMin": str(use["code_runs_min"]), "CodeRunsMax": str(use["code_runs_max"]),
        "OutputTokensMean": thousands(round(use["output_tokens_mean"])),
        "ThinkingTokensMean": thousands(round(use["thinking_tokens_mean"])),
        "InputTokensMean": thousands(round(use["input_tokens_mean"], -2)),
        "CacheReadPct": f"{100 * use['cache_read_share']:.0f}",
        "CacheWriteCostPct": f"{100 * use['cache_write_cost_share']:.0f}",
        "DevCostTotal": f2(sum(dev_usd.values())),
        "ProjectCostTotal": f2(sum(dev_usd.values()) + usage["usd_total_all_attempts"]),
        # development
        "DevVTwo": f3(dev["v2_500"]["macro_auroc"]),
        "DevVTwoLow": f3(dev["v2_500"]["macro_auroc_ci95"][0]), "DevVTwoHigh": f3(dev["v2_500"]["macro_auroc_ci95"][1]),
        "DevVTwoHundred": f3(dev["v2_100"]["macro_auroc"]),
        "DevBoth": f3(dev["both_100"]["macro_auroc"]),
        "DevBothDiff": s3(dev["both_vs_v2_100"]["diff"]),
        "DevBothDiffLow": s3(dev["both_vs_v2_100"]["ci95"][0]),
        "DevBothDiffHigh": s3(dev["both_vs_v2_100"]["ci95"][1]),
        "DevBothDiffHalfWidth": f3(half_width(dev["both_vs_v2_100"]["ci95"])),
        "DevStatements": f3(dev["statements_100"]["macro_auroc"]),
        "DevStatementsDiff": s3(dev["statements_vs_v2_100"]["diff"]),
        "DevStatementsDiffLow": s3(dev["statements_vs_v2_100"]["ci95"][0]),
        "DevStatementsDiffHigh": s3(dev["statements_vs_v2_100"]["ci95"][1]),
        "DevStatementsDiffHalfWidth": f3(half_width(dev["statements_vs_v2_100"]["ci95"])),
        "DevFreshVTwo": f3(dev["v2_200"]["macro_auroc"]),
        "DevFreshVTwoLow": f3(dev["v2_200"]["macro_auroc_ci95"][0]),
        "DevFreshVTwoHigh": f3(dev["v2_200"]["macro_auroc_ci95"][1]),
        "DevFreshVFive": f3(dev["v5_200"]["macro_auroc"]),
        "DevFreshVFiveDiff": s3(dev["v5_vs_v2_200"]["diff"]),
        "DevFreshVFiveDiffLow": s3(dev["v5_vs_v2_200"]["ci95"][0]),
        "DevFreshVFiveDiffHigh": s3(dev["v5_vs_v2_200"]["ci95"][1]),
        "DevFreshVFiveDiffHalfWidth": f3(half_width(dev["v5_vs_v2_200"]["ci95"])),
        "DevAgeSexFiveHundred": f3(dev["agesex_500"]["macro_auroc"]),
        "DevAgeSexTwoHundred": f3(dev["agesex_200"]["macro_auroc"]),
        "DevVTwoAll": f3(dev["v2_700"]["macro_auroc"]),
        "DevVTwoAllLow": f3(dev["v2_700"]["macro_auroc_ci95"][0]),
        "DevVTwoAllHigh": f3(dev["v2_700"]["macro_auroc_ci95"][1]),
        "NDevVTwoAll": thousands(len(ids700)),
        "DevAgeSexAll": f3(dev["agesex_700"]["macro_auroc"]),
        "DevAgeSexAllLow": f3(dev["agesex_700"]["macro_auroc_ci95"][0]),
        "DevAgeSexAllHigh": f3(dev["agesex_700"]["macro_auroc_ci95"][1]),
        "NDevRecordsUsed": thousands(len(dev_ids)),
        "NPilot": str(len(pilot)), "NPilotInMain": str(len(pilot & set(dev500))),
        "NPilotExtra": str(len(pilot - set(dev500) - set(ids200))),
        "NDevFreshAndMain": thousands(len(set(dev500) | set(ids200))),
        "DevAgeSexFold": f3(summary(preds(AGESEX_DEV))["macro_auroc"]),
        "NValFold": thousands(len(preds(AGESEX_DEV))),
        "NWithSuperclass": thousands(int((m["labels"].str.len() > 0).sum())),
        "NDevRuns": str(len(DEV_RUNS)),
    }
    # published comparators (benchmarks/ptbxl_superdiagnostic_published.csv), quoted as reported
    from ecgcode.evaluate import load_benchmark_table
    bench = load_benchmark_table(ROOT / "benchmarks" / "ptbxl_superdiagnostic_published.csv")
    g = lambda grp: bench[bench["group"] == grp]["macro_auroc"]  # noqa: E731
    same = bench[bench["same_protocol"] == "yes"]
    best_same = float(same["macro_auroc"].max())
    mac.update({
        "BestPublishedAUROC": f3(best_same),
        "MaxReportedAUROC": f3(bench[bench["group"].isin(["benchmark", "supervised", "pretrained"])]["macro_auroc"].max()),
        "MinLaterAUROC": f3(bench[bench["group"].isin(["supervised", "pretrained"])]["macro_auroc"].min()),
        "BestSingleBenchmarkAUROC": f3(bench[(bench["group"] == "benchmark") &
                                             ~bench["model"].str.contains("ensemble")]["macro_auroc"].max()),
        "BestZeroShotAUROC": f3(g("zero_shot").max()), "MinZeroShotAUROC": f3(g("zero_shot").min()),
        "BestECGLLMAUROC": f3(g("ecg_llm").max()), "MinECGLLMAUROC": f3(g("ecg_llm").min()),
        "BestGeneralLLMAUROC": f3(g("general_llm").max()), "MinGeneralLLMAUROC": f3(g("general_llm").min()),
        "GapToBestPublished": s3(primary["macro_auroc"] - best_same),
        "GapToBestZeroShot": s3(primary["macro_auroc"] - float(g("zero_shot").max())),
        "GapToBestPublishedAbs": f3(best_same - primary["macro_auroc"]),
        "NOtherTestSet": "2{,}082",
    })
    for label in LABELS:
        mac[f"Main{label}"] = f3(ext["per_class_auroc"][label])
        mac[f"Main{label}Low"] = f3(ext["per_class_ci95"][label][0])
        mac[f"Main{label}High"] = f3(ext["per_class_ci95"][label][1])
        mac[f"AgeSex{label}"] = f3(agesex_sum["per_class_auroc"][label])
        mac[f"CodeVsAgeSex{label}"] = s3(vs_agesex_pc[label]["diff"])
        mac[f"CodeVsAgeSex{label}Low"] = s3(vs_agesex_pc[label]["ci95"][0])
        mac[f"CodeVsAgeSex{label}High"] = s3(vs_agesex_pc[label]["ci95"][1])
        mac[f"Pos{label}"] = thousands(ext["positives"][label])
        mac[f"Prev{label}"] = f"{100 * ext['positives'][label] / len(ids):.1f}"
        mac[f"AUPRC{label}"] = f3(ext["per_class_auprc"][label])
    write_macros(GEN / "results.tex", mac, "python analysis/results.py (pre-specified endpoints and development)")
    write_json({"primary": primary, "per_class_ci": ext, "vs_agesex": vs_agesex,
                "vs_agesex_per_class": vs_agesex_pc, "agesex": agesex_sum, "usage": use, "usage_json": usage,
                "development": dev, "dev_usd": dev_usd},
               OUT_JSON / "results.json")
    print(f"primary {mac['MainAUROC']} [{mac['MainAUROCLow']}, {mac['MainAUROCHigh']}]; "
          f"vs age-sex {mac['CodeVsAgeSex']}; dev v2 on 500 {mac['DevVTwo']}; "
          f"v2 on 700 {mac['DevVTwoAll']}; age-sex on 700 {mac['DevAgeSexAll']}")
    for k in ["both_vs_v2_100", "statements_vs_v2_100", "v5_vs_v2_200"]:
        print(k, round(dev[k]["a"], 3), round(dev[k]["b"], 3), round(dev[k]["diff"], 3), [round(x, 3) for x in dev[k]["ci95"]])
    print("usage", json.dumps(use))


if __name__ == "__main__":
    main()
