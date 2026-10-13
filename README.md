# LLM with a Python sandbox reading raw 12-lead ECGs (PTB-XL)

A general-purpose LLM (`claude-opus-5-5`) receives each ECG as a raw file: all 12 leads, the full 10 seconds at
500 Hz, in millivolts, plus age and sex. In a Python sandbox (the API's code-execution tool, with numpy/scipy/pandas
and no internet) it measures the recording itself, then gives a probability for each of the five PTB-XL diagnostic
superclasses. There is no training, fine-tuning or retrieval, and the prompt contains no labelled examples or class
prevalences. The configuration was developed on PTB-XL fold 9 and evaluated once on the official test fold (fold 10);
folds 1–8 were used only for the age-and-sex control and the fallback class prevalences.

**Result (pre-specified endpoints):** macro AUROC 0.886 (95% CI 0.878–0.895) on all 2,158 test records, exceeding
an age-and-sex control by +0.200, at USD 0.043 per ECG (batch prices).

## Layout

| Path | What |
|---|---|
| `ecgcode/` | the pipeline (frozen): data and labels, record files, requests, API calls, parsing, evaluation |
| `config/default.toml`, `prompts/` | frozen configuration and prompts (`reader_code_v2_measure.md` is the frozen prompt) |
| `requirements.txt`, `requirements-lock.txt` | dependencies (frozen, not pinned) and the exact versions tested |
| `protocol/` | record-id lists of the validation samples used in development |
| `controls/` | predictions of the age-and-sex control (`analysis/agesex_control.py`) |
| `benchmarks/` | published PTB-XL superclass results with their sources |
| `analysis/` | scripts that produce all tables, figures and results of the paper (offline) |
| `paper/` | the manuscript (`main.tex`), the Supplementary Information (`supplement.tex`), figures |
| `scripts/` | `test_run.py`, the driver of the test run (paid) |
| `tests/` | offline tests (no API calls) |
| `runs/` | run folders with requests, transcripts, answers, token counts and costs (git-ignored; attached as `runs.tar.gz` to release v1.0) |

## Setup

Python 3.11–3.13 (tested with 3.13.9). `requirements-lock.txt` pins pandas 2.2.3, which has no wheels for
Python 3.14 or later; on a newer Python, install `requirements.txt` instead (versions not tested).

```bash
pip install -r requirements-lock.txt   # exact tested versions; the frozen requirements.txt is not pinned
mkdir -p data && ln -s /path/to/ptb-xl data/ptb-xl   # PTB-XL 1.0.3 from PhysioNet
python -m pytest -q tests
export ANTHROPIC_API_KEY=...   # never committed
```

## Commands

| Command | API | Cost |
|---|---|---|
| `build`, `estimate`, `parse`, `evaluate`, `spend` | none | free |
| `upload`, `delete-files`, `preflight`, `collect`, `reconcile`, `cancel` | Files API, batches, count_tokens | free (`collect` pays only for continuations after `pause_turn`) |
| `submit` (Message Batches, 50% price; `--sync` one at a time) | Messages API with the code-execution tool | **paid** |

A request with the code-execution tool re-reads its growing context after every sandbox round, so its cost is not
bounded by `max_tokens`. `submit` checks the total cap (`spend --set-cap`) against an assumed worst case per request,
logs every paid result in the spend ledger (`runs/spend_ledger.jsonl`), and refuses a test-fold run unless it was
built with the freeze check passed. `build --split test` refuses unless the code, configuration and prompts are
unchanged since the freeze and no experimental setting other than the record-shuffle seed (`run.seed`) differs from the
frozen configuration; the test run used the frozen seed (7).

## Test run (as made)

```bash
python -m ecgcode build --split test --run runs/test_v2        # refuses unless frozen
python scripts/test_run.py                                      # PAID: upload, then submit + collect in
                                                                # batches of 250, stopping on cost or failures
```

## Reproduce the paper (offline)

1. PTB-XL 1.0.3 in `data/ptb-xl` (Setup).
2. The run folders: download `runs.tar.gz` from release v1.0 of this repository and unpack it in the repository root
   (`tar -xzf runs.tar.gz` gives `runs/`, with the folder names the scripts use: `test_v2`, `dev500`, `dev200b_v2`
   and so on). The scripts read `requests.jsonl`, `results.jsonl`, `predictions.csv`, `key.json`, `usage.json`, `batches.json` (runs
   sent as batches) and `manifest.json` from the run folders. `key.json` maps the anonymous record ids to PTB-XL
   `ecg_id`s; during the study it was stored locally and was never part of any request in the development or test
   runs. The record files in `files/` are needed only by `analysis/replay.py`; `analysis/si.py` rebuilds the one it
   needs from PTB-XL.
3. Run:

```bash
python analysis/agesex_control.py  # the age-and-sex control (controls/)
python -m ecgcode evaluate --run runs/test_v2 --against controls/agesex_test   # optional: the primary endpoint and
                                   # the paired difference from the control, printed by the frozen evaluation code
python analysis/results.py     # pre-specified endpoints (frozen evaluation code) and development table
python analysis/posthoc.py     # post hoc analyses and supplementary tables
python analysis/si.py          # prompt, schema, example message and transcript for the SI
python analysis/figures.py     # figures
python analysis/replay.py      # post hoc: re-runs the model's code offline (below); about 20 min on 8 cores
cd paper && make               # main.pdf and supplement.pdf; or: conda run -n tex make (Tectonic)
```

`analysis/replay.py` executes every code run of the test run again, in a sandbox without network access, and compares
the printed output with what the API's sandbox returned. It needs bubblewrap (`bwrap`; `sudo apt install bubblewrap`)
and a separate Python matching the API container, by default the conda environment `replay` (otherwise pass
`--python-prefix`):

```bash
conda create -n replay python=3.11.17
conda run -n replay pip install numpy==2.2.6 scipy==1.15.3 pandas==2.2.3 matplotlib==3.10.6
```

## Licence

Code: MIT (`LICENSE`). PTB-XL is not included; it is available from PhysioNet under CC BY 4.0. Some files here
contain data derived from it: the age-and-sex control predictions (`controls/`), the PTB-XL record ids in `protocol/`, and the example record shown in Fig. 1 and the Supplementary Information
(`paper/figures/`, `paper/generated/`). The run archive (`runs.tar.gz`,
release v1.0) contains the record files the model received, derived from PTB-XL, and is released under CC BY 4.0.
Cite PTB-XL (Wagner et al., Sci Data 2020) when reusing them.
