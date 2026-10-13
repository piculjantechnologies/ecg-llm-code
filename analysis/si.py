"""Supplementary Information inputs, generated from the frozen files and the run folders (no API calls).

Run from the repository root:  python analysis/si.py
Writes to paper/generated/:
  prompt.tex           the frozen system prompt, verbatim (ASCII transliteration of a few symbols)
  schema.tex           the output schema sent with every request
  example_input.tex    the user message of the Fig. 1 record and the first rows of its file
  transcript.tex       the complete sandbox transcript of the Fig. 1 record (code, printed output,
                       summarized thinking as returned by the API, answer)
  table_benchmark.tex  published results (benchmarks/ptbxl_superdiagnostic_published.csv) and this work
"""
import json
import textwrap

import re

from common import GEN, ROOT, RUNS, meta, write_macros
from ecgcode.data import load_signal
from ecgcode.evaluate import load_benchmark_table
from ecgcode.prompting import output_schema
from ecgcode.records import record_csv
from figures import code_blocks, example, user_message

ASCII = {"\u2192": "->", "\u2014": "--", "\u2013": "-", "\u2212": "-", "\u2265": ">=", "\u2264": "<=",
         "\u00b0": " deg", "\u00d7": "x", "\u2248": "~", "\u00b1": "+/-", "\u2019": "'", "\u2018": "'",
         "\u201c": '"', "\u201d": '"', "\u2026": "...", "\u00b5": "u", "\u2032": "'"}
GROUP_TITLES = {
    "benchmark": "Original PTB-XL benchmark: supervised, trained on folds 1--8",
    "supervised": "Later supervised models",
    "pretrained": "Pretrained on large ECG corpora, then fine-tuned on PTB-XL",
    "zero_shot": "Zero-shot ECG--language models (encoder pretrained on ECGs with reports or report-derived labels)",
    "ecg_llm": "Multimodal LLMs instruction-tuned on ECG data",
    "general_llm": "General-purpose multimodal LLMs, zero-shot, ECG images",
}


def ascii_(text):
    for k, v in ASCII.items():
        text = text.replace(k, v)
    bad = sorted({c for c in text if ord(c) > 127})
    if bad:
        raise ValueError(f"non-ASCII characters left: {bad}")
    return text


def wrap(text, width=88):
    """Lines longer than width are broken at spaces, and every continuation line is indented by exactly two spaces:
    the space at a break stays at the end of the line before it (invisible), so deleting each line break together
    with the two spaces after it restores the text exactly."""
    out = []
    for line in text.splitlines():
        if len(line) <= width:
            out.append(line)
            continue
        parts = textwrap.fill(line, width, subsequent_indent="  ", break_on_hyphens=False, drop_whitespace=False,
                              replace_whitespace=False).split("\n")
        lines = [parts[0]]
        for p in parts[1:]:
            body = p[2:]                                  # without the indent
            space = body[:len(body) - len(body.lstrip(" "))]
            lines[-1] += space                            # the break space(s) end the line before
            if body[len(space):]:
                lines.append("  " + body[len(space):])
        out.extend(lines)
    return "\n".join(out)


def verbatim(text, size="small", width=88):
    return f"\\begin{{{size}}}\n\\begin{{verbatim}}\n{wrap(ascii_(text), width)}\n\\end{{verbatim}}\n\\end{{{size}}}\n"


def tex(text):
    return text.replace("_", "\\_").replace("&", "\\&").replace("%", "\\%")


def record_file(run, anon, ecg_id):
    """The CSV file the model received. A deposit of the run folders may leave out files/; the file is then
    rebuilt from PTB-XL with the frozen record writer and the run's input settings (identical bytes)."""
    path = RUNS / run / "files" / f"{anon}.csv"
    if path.exists():
        return path.read_text()
    settings = json.loads((RUNS / run / "manifest.json").read_text())["config"]["input"]
    fs = int(settings["sampling_rate"])
    return record_csv(load_signal(meta(), ecg_id, fs), fs, int(settings["decimals"]))


def main():
    manifest = json.loads((RUNS / "test_v2" / "manifest.json").read_text())
    prompt_path = ROOT / manifest["prompt_file"]
    text = prompt_path.read_text()
    (GEN / "prompt.tex").write_text(f"% Generated from {manifest['prompt_file']} (sha256 {manifest['prompt_sha256']})\n"
                                    + verbatim(text.strip()))
    (GEN / "schema.tex").write_text(verbatim(json.dumps(output_schema(), indent=1)))

    ecg_id, anon, run, r, pred = example()
    msg = user_message(run, anon)
    rows = record_file(run, anon, ecg_id).splitlines()
    # the PTB-XL recording may begin with identical samples; say so, so that the repeated rows are not taken for an
    # error of the file writer
    n_same = next(i for i in range(1, len(rows)) if rows[i].split(",", 1)[1] != rows[1].split(",", 1)[1]) - 1
    same = f"\n(The first {n_same} samples are identical, as in the PTB-XL source file.)" if n_same > 1 else ""
    example_txt = (f"User message (text block):\n{msg}\n\n"
                   f"Attached file (container_upload block): {anon}.csv, {len(rows) - 1:,} data rows. First rows:\n"
                   + "\n".join(rows[:4]) + "\n..." + same)
    (GEN / "example_input.tex").write_text(verbatim(example_txt, "footnotesize", 99))   # 99 characters fit the page

    parts = []
    for i, b in enumerate(code_blocks(r), 1):
        out = b["out"].rstrip()
        out_lines = out.splitlines()
        if len(out_lines) > 30:
            out = "\n".join(out_lines[:30] + [f"[... {len(out_lines) - 30} more lines]"])
        parts.append(f"=== Code run {i}: command ===\n{b['code'].strip()}\n=== Code run {i}: printed output ===\n{out}")
    answer = [b["text"] for b in r["content"] if b.get("type") == "text"][-1]
    parts.append("=== Summarized thinking (as returned by the API; the full thinking is not returned) ===\n"
                 + (r.get("thinking_summary") or "").strip())
    parts.append("=== Final answer ===\n" + json.dumps(json.loads(answer), indent=1, ensure_ascii=False))
    (GEN / "transcript.tex").write_text(
        f"% Generated from {run}/results.jsonl, record {anon} (PTB-XL ecg_id {ecg_id})\n"
        + verbatim("\n\n".join(parts), "scriptsize", 104))

    # macros for the text about the example record (Fig. 1)
    blocks = code_blocks(r)
    printed = "\n".join(b_["out"] for b_ in blocks)
    ax_printed = re.findall(r"^axis\s+(-?\d+\.?\d*)", printed, re.M)
    qrs_printed = re.findall(r"QRS ms\s+(\d+)", printed)
    kf = pred["key_findings"]
    ax_stated = re.search(r"axis[^.;]*?([+-]?\d{1,3})\s*\u00b0", kf)
    m = meta()
    write_macros(GEN / "example.tex", {
        "ExampleECGID": str(ecg_id), "ExampleAnon": "\\texttt{" + anon + "}", "ExampleCodeRuns": str(len(blocks)),
        "ExampleAge": str(int(m.loc[ecg_id, "age"])),
        "ExampleAxisPrinted": f"{float(ax_printed[-1]):.1f}" if ax_printed else "--",
        "ExampleAxisStated": ax_stated.group(1).replace("+", "") if ax_stated else "--",
        "ExampleQRSPrinted": qrs_printed[-1] if qrs_printed else "--",
        "ExamplePHYP": f"{pred['p_HYP']:.2f}", "ExamplePNORM": f"{pred['p_NORM']:.2f}",
        "ExampleUSD": f"{r['usd']:.3f}",
    }, "python analysis/si.py (the Fig. 1 example record)")

    bench = load_benchmark_table(ROOT / "benchmarks" / "ptbxl_superdiagnostic_published.csv")
    lines = []
    for group, title in GROUP_TITLES.items():
        rows_ = bench[bench["group"] == group].sort_values("macro_auroc", ascending=False)
        lines.append(f"\\multicolumn{{3}}{{l}}{{\\emph{{{title}}}}} \\\\")
        for x in rows_.itertuples():
            unc = f" ({tex(x.reported_uncertainty)})" if isinstance(x.reported_uncertainty, str) else ""
            mark = "" if x.same_protocol == "yes" else "$^\\dagger$"
            lines.append(f"  \\quad {tex(x.model)}~\\citep{{{x.citekey}}}{mark} & {tex(x.input)} & "
                         f"{x.macro_auroc:.3f}{unc} \\\\")
        lines.append("\\midrule")
    (GEN / "table_benchmark.tex").write_text(
        "% Generated by python analysis/si.py from benchmarks/ptbxl_superdiagnostic_published.csv\n"
        "\\begin{tabular}{L{0.52\\linewidth}L{0.19\\linewidth}l}\n\\toprule\n"
        "Method & Input & Macro AUROC \\\\\n\\midrule\n" + "\n".join(lines) + "\n"
        "\\multicolumn{3}{l}{\\emph{This work (test fold, \\NTest{} records)}} \\\\\n"
        "  \\quad General-purpose LLM with a Python sandbox\\newline\\hspace*{1em}(\\ModelID) & "
        "raw signal (500 Hz), age, sex & \\MainAUROC{} [\\MainAUROCLow, \\MainAUROCHigh] \\\\\n"
        "  \\quad Age and sex only (control, logistic regression) & demographics & "
        "\\AgeSexAUROC{} [\\AgeSexAUROCLow, \\AgeSexAUROCHigh] \\\\\n"
        "\\bottomrule\n\\end{tabular}\n")
    print("wrote prompt, schema, example_input, transcript, table_benchmark for record", anon, ecg_id)


if __name__ == "__main__":
    main()
