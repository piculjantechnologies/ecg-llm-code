"""Command line: python -m ecgcode <command> (run from the repository root).

Paid: submit. Free API calls: upload, delete-files, preflight. Everything else is offline.
"""
import argparse
import json
from pathlib import Path

from . import api, evaluate, runs, spend
from .config import load_config, project_path


def _cfg(args):
    return load_config(args.config, args.set)


def cmd_build(args):
    exclude = runs.read_ids(args.exclude_ids) if args.exclude_ids else []
    for other in args.exclude_runs:
        exclude += runs.load_manifest(other)["ecg_ids"]
    m = runs.build_run(_cfg(args), args.run, args.split, n=args.n, sample_seed=args.sample_seed,
                       include_ids=runs.read_ids(args.include_ids) if args.include_ids else None,
                       exclude_ids=exclude or None, allow_unfrozen=args.allow_unfrozen)
    git = m["git"]
    print(f"Built {args.run}: {m['n_records']} records ({m['split']}); prompt sha256 {m['prompt_sha256'][:12]}; "
          f"code {str(git.get('commit'))[:10]}{' (UNCOMMITTED CHANGES)' if git.get('dirty') else ''}")
    print("  settings changed from config/default.toml: " + ("; ".join(m["config_diff_from_default"]) or "none"))


def cmd_upload(args):
    with runs.run_lock(args.run):
        api.upload_files(args.run)


def cmd_delete_files(args):
    with runs.run_lock(args.run):
        api.delete_files(args.run)


def cmd_preflight(args):
    api.preflight(args.run)


def cmd_estimate(args):
    todo = api.pending_requests(args.run, limit=args.limit)
    print(json.dumps(api.plan_cost(args.run, todo, _cfg(args)["cost"], "sync"), indent=2))


def cmd_submit(args):
    with spend.submit_lock(), runs.run_lock(args.run):
        cfg = _cfg(args)
        manifest = runs.load_manifest(args.run)
        if manifest["split"] == "test" and not manifest.get("frozen_build"):
            raise SystemExit("Refused: this test-fold run was built without the protocol freeze.")
        api.guard_inflight(args.run)
        todo = api.pending_requests(args.run, limit=args.limit)
        if todo and not args.sync:
            return _submit_batch(args, cfg, manifest, todo)
        if not todo:
            print("Nothing to submit.")
            return
        plan = api.plan_cost(args.run, todo, cfg["cost"], "sync")
        worst_one = plan["usd_worst_per_request"]
        if args.max_usd < worst_one:
            raise SystemExit(f"Refused: --max-usd {args.max_usd:.2f} is below the assumed worst case of one "
                             f"request (${worst_one:.2f}).")
        spent_total = spend.spent_total()
        cap = spend.total_cap(cfg["budget"])
        if spent_total + args.max_usd > cap:
            raise SystemExit(f"Refused: spent ${spent_total:.2f} + --max-usd {args.max_usd:.2f} would exceed the total "
                             f"cap ${cap:.2f} (python -m ecgcode spend --set-cap USD).")
        mc = manifest["config"]["model"]
        spend.confirm([
            f"About to send up to {len(todo)} request(s) from {args.run} (sync, full price, one at a time).",
            f"  model {mc['id']}, effort {mc['effort']}, max_tokens {mc['max_tokens']}, tool {mc['code_tool']}",
            f"  prices: ${cfg['cost']['input']}/${cfg['cost']['output']} per MTok (input/output)",
            f"  expected ${plan['usd_expected']:.2f} ({'measured' if plan['calibration'] else 'rough guess'}); "
            f"sending stops once actual cost reaches ${args.max_usd - worst_one:.2f} (--max-usd {args.max_usd:.2f} "
            f"minus one assumed worst request ${worst_one:.2f})",
            f"  already spent (this project's ledger) ${spent_total:.2f} of the ${cap:.2f} cap",
        ], token=Path(args.run).name, assume_yes=args.yes)
        api.run_sync(args.run, todo, cfg["cost"], args.max_usd, int(mc["max_continuations"]), worst_one)
        print(json.dumps(runs.parse_run(args.run), indent=2))


def _submit_batch(args, cfg, manifest, todo):
    plan = api.plan_cost(args.run, todo, cfg["cost"], "batch")
    committed = api.committed_usd(api.all_run_dirs([args.run]))
    planned, spent_total = spend.check_budget(plan, cfg["budget"], args.max_usd, args.budget_basis, committed)
    mc = manifest["config"]["model"]
    spend.confirm([
        f"About to send {len(todo)} request(s) from {args.run} as a Message Batch (50% price).",
        f"  model {mc['id']}, effort {mc['effort']}, max_tokens {mc['max_tokens']}, tool {mc['code_tool']}",
        f"  expected ${plan['usd_expected']:.2f} "
        f"({'measured on %d earlier requests' % plan['calibration']['n'] if plan['calibration'] else 'rough guess'}); "
        f"assumed worst case ${plan['usd_worst_case']:.2f}",
        f"  basis '{args.budget_basis}' -> ${planned:.2f} <= --max-usd {args.max_usd:.2f}; spent ${spent_total:.2f} "
        f"+ running batches ${committed:.2f} + this worst case <= cap ${spend.total_cap(cfg['budget']):.2f}",
    ], token=Path(args.run).name, assume_yes=args.yes)
    api.submit_batches(args.run, todo, plan["usd_worst_per_request"], cfg["cost"])
    print(f"Now: python -m ecgcode collect --run {args.run}")


def cmd_collect(args):
    cfg = _cfg(args)
    if not args.no_wait:
        api.wait_for_batches(args.run, poll_seconds=args.poll_seconds)
    with runs.run_lock(args.run):
        mc = runs.load_manifest(args.run)["config"]["model"]
        api.collect_batches(args.run, cfg["cost"], int(mc["max_continuations"]))
        print(json.dumps(runs.parse_run(args.run), indent=2))


def cmd_reconcile(args):
    with runs.run_lock(args.run):
        api.reconcile(args.run, attach=args.attach, drop_unknown=args.drop_unknown)


def cmd_cancel(args):
    api.cancel_batches(args.run)


def cmd_parse(args):
    print(json.dumps(runs.parse_run(args.run), indent=2))


def cmd_evaluate(args):
    cfg = runs.load_manifest(args.run)["config"]
    result = evaluate.evaluate_run(args.run, cfg)
    if args.against:
        result["paired_vs"] = evaluate.paired_difference(args.run, args.against, cfg,
                                                          allow_subset=args.allow_subset)
    print(json.dumps(result, indent=2))


def cmd_spend(args):
    if args.set_cap is not None:
        spend.set_cap(args.set_cap)
        print(f"Total cap set to ${args.set_cap:.2f} ({spend.cap_path()})")
    for run, usd in sorted(spend.spent_by_run().items()):
        print(f"  ${usd:9.2f}  {run}")
    print(f"total ${spend.spent_total():.2f} of cap ${spend.total_cap(_cfg(args)['budget']):.2f} "
          f"(ledger {spend.ledger_path()})")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m ecgcode")
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build", help="select records, write their raw CSV files and requests (offline)")
    p.add_argument("--run", required=True)
    p.add_argument("--split", choices=list(runs.SPLITS), required=True)
    p.add_argument("--n", type=int)
    p.add_argument("--sample-seed", type=int, default=0)
    p.add_argument("--include-ids")
    p.add_argument("--exclude-ids")
    p.add_argument("--exclude-runs", nargs="+", default=[])
    p.add_argument("--allow-unfrozen", action="store_true", help="test-fold dry runs only; never submittable")
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("upload", help="FREE: upload the run's record files to the Files API")
    p.add_argument("--run", required=True)
    p.set_defaults(func=cmd_upload)

    p = sub.add_parser("delete-files", help="FREE: delete the run's uploaded record files")
    p.add_argument("--run", required=True)
    p.set_defaults(func=cmd_delete_files)

    p = sub.add_parser("preflight", help="FREE: credentials, model, request shape")
    p.add_argument("--run", required=True)
    p.set_defaults(func=cmd_preflight)

    p = sub.add_parser("estimate", help="expected and assumed worst cost of what submit would send (offline)")
    p.add_argument("--run", required=True)
    p.add_argument("--limit", type=int)
    p.set_defaults(func=cmd_estimate)

    p = sub.add_parser("submit", help="PAID: send pending requests (batch by default; --sync one at a time)")
    p.add_argument("--run", required=True)
    p.add_argument("--max-usd", type=float, required=True,
                   help="batch: refuse if the planned cost exceeds it; sync: stop sending before it is passed")
    p.add_argument("--budget-basis", choices=["worst", "expected"], default="worst")
    p.add_argument("--sync", action="store_true")
    p.add_argument("--limit", type=int)
    p.add_argument("--yes", action="store_true")
    p.set_defaults(func=cmd_submit)

    p = sub.add_parser("collect", help="FREE: wait for batches, download, resume paused requests (paid), parse")
    p.add_argument("--run", required=True)
    p.add_argument("--no-wait", action="store_true")
    p.add_argument("--poll-seconds", type=int, default=60)
    p.set_defaults(func=cmd_collect)

    p = sub.add_parser("reconcile", help="FREE: resolve a batch creation with an unknown outcome")
    p.add_argument("--run", required=True)
    p.add_argument("--attach", metavar="BATCH_ID")
    p.add_argument("--drop-unknown", action="store_true")
    p.set_defaults(func=cmd_reconcile)

    p = sub.add_parser("cancel", help="cancel the run's uncollected batches")
    p.add_argument("--run", required=True)
    p.set_defaults(func=cmd_cancel)

    p = sub.add_parser("parse", help="rebuild predictions.csv, failures.csv, usage.json")
    p.add_argument("--run", required=True)
    p.set_defaults(func=cmd_parse)

    p = sub.add_parser("evaluate", help="macro AUROC with bootstrap CI")
    p.add_argument("--run", required=True)
    p.add_argument("--against")
    p.add_argument("--allow-subset", action="store_true")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("spend", help="money spent (this project's ledger); --set-cap to change the cap")
    p.add_argument("--set-cap", type=float)
    p.set_defaults(func=cmd_spend)

    args = parser.parse_args(argv)
    try:
        args.func(args)
    except ValueError as e:
        raise SystemExit(f"error: {e}")
