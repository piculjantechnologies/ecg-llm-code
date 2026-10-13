"""Publication figures (no API calls; reads run folders, controls/, PTB-XL and analysis/*.json).

Run from the repository root, after analysis/results.py and analysis/posthoc.py:

    python analysis/figures.py            # all figures
    python analysis/figures.py fig1 figS1 # a subset

Writes vector PDFs to paper/figures/:
  fig1_example.pdf    one validation-fold record: the raw file, the model's code, its printed output and answer
  fig2_benchmark.pdf  a: macro AUROC among published PTB-XL superclass results; b: per-class AUROC
  fig3_where.pdf      a: AUROC by diagnostic statement; b: stated measurements vs PTB-XL statements
  figS1_curves.pdf    ROC, precision-recall and reliability curves of the test run (Supplementary Fig. 2)
  figS2_usage.pdf     code runs and cost per record of the test run (Supplementary Fig. 1)

The Supplementary Information numbers its figures in the order of first citation, so the file names figS1 and figS2
do not follow its numbering.

The example record in Fig. 1 is chosen by a fixed rule (EXAMPLE_RULE), not by its result. Sizes follow
Nature Portfolio column widths (89 mm single, 183 mm double).
"""
import json
import re
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import transforms  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (LABELS, PROB_COLS, ROOT, RUNS, TEST, custom_to_anon, key, meta, preds,  # noqa: E402
                    results_lines, truth)
from posthoc import calib_bins  # noqa: E402
from ecgcode.data import load_signal  # noqa: E402
from ecgcode.evaluate import load_benchmark_table  # noqa: E402

OUT = ROOT / "paper" / "figures"
MM = 1 / 25.4
SINGLE, DOUBLE = 89 * MM, 183 * MM

CLASS_COLOURS = dict(zip(LABELS, ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"]))
CODE = "#0072B2"      # this study: the model
INK, MUTED, LIGHT = "#222222", "#6E6E6E", "#BDBDBD"
V2_RUNS = ["dev100_measure", "dev400_measure", "dev200b_v2"]
EXAMPLE_RULE = ("the validation-fold record with the lowest PTB-XL ecg_id whose only superclass is HYP, "
                "among the records read with the frozen prompt during development")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.monospace": ["DejaVu Sans Mono", "Liberation Mono"],
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "text.color": INK,
    "legend.frameon": False, "lines.linewidth": 1.0,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.dpi": 600,
})


def panel_label(ax, letter, x=-0.02, y=1.0):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="right")


def below_top(ax, pt):
    """Axes coordinates shifted down by pt points: text placed with it keeps its distance from the top edge of the
    axes when the axes height changes."""
    return transforms.offset_copy(ax.transAxes, fig=ax.figure, y=-pt, units="points")


TITLE_PAD = 3  # points between an axes and its title (set_title pad)


def title_label(ax, letter, x_fig):
    """Panel letter on the baseline of the axes title, its right edge at x_fig (figure fraction)."""
    x0, _, w, _ = ax.get_position().bounds
    ax.text((x_fig - x0) / w, 1.0, letter, transform=below_top(ax, -TITLE_PAD), fontsize=9, fontweight="bold",
            va="baseline", ha="right")


def save(fig, name, pad_inches=0.02):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=pad_inches, metadata={"CreationDate": None})
    plt.close(fig)
    print(f"wrote {OUT / name}")


def minus(s):
    return s.replace("-", "−")


def tex_text(s):
    """A LaTeX criterion string from posthoc_results.json as figure text (true minus, ≥, ≤, °, en dash)."""
    s = re.sub(r"\$([^$]*)\$", lambda mm: minus(mm.group(1)), s)  # math segments: minus signs
    for a, b in [("^{\\circ}", "°"), ("^\\circ", "°"), ("\\geq", "≥"), ("\\leq", "≤"), ("\\ge", "≥"),
                 ("\\le", "≤"), ("\\,", " "), ("~", " "), ("\\%", "%"), ("--", "–"),
                 ("Sokolow-Lyon", "Sokolow–Lyon")]:
        s = s.replace(a, b)
    if any(c in s for c in "\\${}"):
        raise ValueError(f"LaTeX left in figure label: {s!r}")
    return " ".join(s.split())


# --- Figure 1: one record, from file to answer ------------------------------------------------------

def example():
    m = meta()
    v2 = preds(*[RUNS / r for r in V2_RUNS])
    ecg_id = min(i for i in v2.index if m.loc[i, "labels"] == ["HYP"])
    for run in V2_RUNS:
        inv = {v: a for a, v in key(RUNS / run).items()}
        if ecg_id not in inv:
            continue
        c2a = custom_to_anon(RUNS / run)
        for r in results_lines(RUNS / run):
            if c2a.get(r["custom_id"]) == inv[ecg_id] and r.get("type") == "succeeded":
                return ecg_id, inv[ecg_id], run, r, v2.loc[ecg_id]
    raise SystemExit("example record not found")


def code_blocks(r):
    blocks = []
    for b in r["content"]:
        if b.get("type") == "server_tool_use":
            blocks.append({"code": b["input"].get("command") or json.dumps(b["input"]), "out": ""})
        elif str(b.get("type", "")).endswith("_tool_result") and blocks:
            blocks[-1]["out"] = str(b.get("content", {}).get("stdout", ""))
    return blocks


def user_message(run, anon):
    """The text of the user message sent with a record, from the run's requests."""
    with open(RUNS / run / "requests.jsonl") as f:
        for line in f:
            q = json.loads(line)
            if q["record_ids"] == [anon]:
                return [b["text"] for b in q["params"]["messages"][0]["content"] if b["type"] == "text"][0]
    raise SystemExit(f"no request for {anon} in {run}")


def fit_box(ax, texts, pad_pt):
    """Move the bottom edge of a boxed text panel (the top stays) to pad_pt points below its lowest text; the texts
    must be placed from the top edge (below_top)."""
    fig = ax.figure
    fig.canvas.draw()
    low = min(t.get_window_extent(fig.canvas.get_renderer()).y0 for t in texts) / fig.bbox.height
    x0, y0, w, h = ax.get_position().bounds
    bottom = low - pad_pt / 72 / fig.get_figheight()
    ax.set_position([x0, bottom, w, y0 + h - bottom])
    return bottom


def fig1_example():
    m = meta()
    ecg_id, anon, run, r, pred = example()
    blocks = code_blocks(r)
    sig = load_signal(m, ecg_id)
    fs = 500
    t = np.arange(len(sig)) / fs
    leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    show = ["II", "V1", "V2", "V5"]

    fig = plt.figure(figsize=(DOUBLE, 150 * MM))  # panels are placed in figure fractions; the bottom is cropped
    H = fig.get_figheight() * 72  # points
    # a: the raw file
    ax = fig.add_axes([0.06, 0.78, 0.885, 0.185])  # room on the right for the 1 mV scale bar
    offs = 0.0
    yt, yl = [], []
    for name in show:
        v = sig[:, leads.index(name)]
        ax.plot(t, v + offs, color=INK, lw=0.45)
        yt.append(offs + np.median(v))
        yl.append(name)
        offs -= 6.0
    ax.set_yticks(yt)
    ax.set_yticklabels(yl)
    ax.set_xlim(0, 10)
    ax.set_xlabel("Time (s)", labelpad=1)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.plot([10.08, 10.08], [yt[0] + 0.5, yt[0] - 0.5], color=INK, lw=1.0, clip_on=False)
    ax.text(10.16, yt[0], "1 mV", va="center", ha="left", clip_on=False)
    msg = user_message(run, anon)
    pt = re.search(r"Patient:[^.]*\.", msg)   # the part of the message that is not the same for every record
    excerpt = ("… " if msg[:pt.start()].strip() else "") + pt.group(0) + (" …" if msg[pt.end():].strip() else "")
    ax.set_title(f"Input: {anon}.csv, {len(sig):,} rows × {len(leads) + 1} columns (t_ms and {len(leads)} leads in mV, "
                 f"raw, {fs} Hz; {len(show)} of {len(leads)} leads shown)\nMessage (excerpt): “{excerpt}”",
                 loc="left", fontsize=8, pad=3, linespacing=1.3)
    left = 0.0175  # right edge of the panel letters of the left column (a, b, d)
    panel_label(ax, "a", x=(left - 0.06) / 0.885, y=1.155)  # level with the first title line

    # b and c: boxes of text placed from the top edge of the box (pad: points above the title); each box then ends
    # just below its text, whose extent already includes the descent and line spacing of the last line
    top, pad = 0.71, 4.6
    # b: code (second code run, the one that forms the median beats and prints them): the whole Python
    # script, without the shell lines around it; long lines cut, marked with "…"
    main_code = blocks[1]["code"]
    body = main_code.split("<<'EOF'\n", 1)[-1].split("\nEOF", 1)[0]
    code_txt = "\n".join(ln[:60] + " …" if len(ln) > 62 else ln for ln in body.splitlines())
    axb = fig.add_axes([0.02, 0.35, 0.50, top - 0.35])
    axb.axis("off")
    axb.add_patch(plt.Rectangle((0, 0), 1, 1, transform=axb.transAxes, fc="#F4F6F8", ec=LIGHT, lw=0.6))
    axb.text(0.02, 1.0, f"Code run 2 of {len(blocks)}: Python script written by the model",
             transform=below_top(axb, pad), va="top", ha="left", fontweight="bold")
    axb.text(0.02, 1.0, "(shell wrapper omitted; long lines cut)", transform=below_top(axb, 12.7), va="top",
             ha="left")
    code_t = axb.text(0.02, 1.0, code_txt, transform=below_top(axb, 22.2), va="top", ha="left", family="monospace",
                      fontsize=6.3, linespacing=1.1)
    panel_label(axb, "b", x=(left - 0.02) / 0.50, y=1.0)

    # c: printed output of the last code run (the measurement table), complete and as printed
    out = blocks[-1]["out"].strip()
    axc = fig.add_axes([0.54, 0.35, 0.44, top - 0.35])
    axc.axis("off")
    axc.add_patch(plt.Rectangle((0, 0), 1, 1, transform=axc.transAxes, fc="#F4F6F8", ec=LIGHT, lw=0.6))
    axc.text(0.03, 1.0, f"Printed output of code run {len(blocks)}", transform=below_top(axc, pad),
             va="top", ha="left", fontweight="bold")
    out_t = axc.text(0.03, 1.0, out, transform=below_top(axc, 15.3), va="top", ha="left", family="monospace",
                     fontsize=6.3, linespacing=1.1)
    panel_label(axc, "c", x=-0.01, y=1.0)
    b_bottom, c_bottom = fit_box(axb, [code_t], 0.5), fit_box(axc, [out_t], 0.5)

    # d and e: the answer, side by side below the longer box (gap: points from it to the top of their titles)
    gap, title_h = 13, TITLE_PAD + 6
    kf = pred["key_findings"]
    row_top = min(b_bottom, c_bottom) - (gap + title_h) / H
    axd = fig.add_axes([0.54, row_top - 0.16, 0.44, 0.16])
    axd.axis("off")
    axd.set_title("Answer (JSON; key findings)", loc="left", fontweight="bold", fontsize=8, pad=TITLE_PAD)
    kf_t = axd.text(0.0, 1.0, textwrap.fill(kf, 70), transform=below_top(axd, 2.6), va="top", ha="left",
                    fontsize=7, linespacing=1.15, style="italic")
    title_label(axd, "e", 0.54 - 0.01 * 0.44)   # as c

    axp = fig.add_axes([0.14, row_top - 0.19, 0.33, 0.19])
    ps = [pred[c] for c in PROB_COLS]
    truth_lab = m.loc[ecg_id, "labels"]
    cols = [CLASS_COLOURS[l] for l in LABELS]
    axp.barh(range(5)[::-1], ps, color=cols, height=0.62)
    axp.set_yticks(range(5)[::-1])
    axp.set_yticklabels([f"{l}{' (label)' if l in truth_lab else ''}" for l in LABELS])
    for i, v in enumerate(ps):
        axp.text(v + 0.015, 4 - i, f"{v:.2f}", va="center", ha="left")
    axp.set_xlim(0, 1)
    axp.set_xticks([0, 0.5, 1])
    axp.set_xlabel("Probability given by the model", labelpad=1)
    axp.set_title("Answer (JSON; probabilities)", loc="left", fontweight="bold", fontsize=8, pad=TITLE_PAD)
    title_label(axp, "d", left)

    usage = r["usage"]
    note = (f"PTB-XL ecg_id {ecg_id} (validation fold)\nStatements: "   # likelihoods written as in the text
            + ", ".join(f"{c} (likelihood {int(v)})" if v else c for c, v in m.loc[ecg_id, "scp_codes"].items())
            + f".\n{len(blocks)} code runs, {usage['output_tokens']:,} output tokens, "
            f"USD {r['usd']:.3f} (batch price).")
    fig.canvas.draw()
    kf_bottom = kf_t.get_window_extent(fig.canvas.get_renderer()).y0 / fig.bbox.height
    fig.text(0.54, kf_bottom - 10 / H, note, ha="left", va="top", color=MUTED, linespacing=1.3)
    save(fig, "fig1_example.pdf")
    return {"ecg_id": ecg_id, "anon": anon, "run": run, "code_runs": len(blocks), "usd": r["usd"],
            "output_tokens": usage["output_tokens"]}


# --- Figure 2: benchmark and per class --------------------------------------------------------------

GROUPS = [  # the sub-headings follow the SI table (analysis/si.py GROUP_TITLES), short enough to end left of the axes
    ("Trained or fine-tuned on PTB-XL (supervised)", [
        ("Original benchmark models, trained on folds 1–8", "benchmark"),
        ("Later supervised models", "supervised"),
        ("Pretrained on ECG corpora, fine-tuned on PTB-XL", "pretrained")]),
    ("Zero-shot ECG–language models", [(None, "zero_shot")]),
    ("Multimodal LLMs", [("Instruction-tuned on ECG data", "ecg_llm"),
                         ("General-purpose, zero-shot, ECG images", "general_llm")]),
]
SHORT_NAMES = {"ensemble of the benchmark models": "Ensemble of the benchmark models"}


def fig2_benchmark():
    res = json.loads((ROOT / "analysis" / "results.json").read_text())
    bench = load_benchmark_table(ROOT / "benchmarks" / "ptbxl_superdiagnostic_published.csv")
    main, ctrl = res["per_class_ci"], res["agesex"]
    unknown = set(bench["group"]) - {g for _, subgroups in GROUPS for _, g in subgroups}
    if unknown:
        raise ValueError(f"benchmark groups without a heading in GROUPS: {sorted(unknown)}")
    rows = []
    for header, subgroups in GROUPS:
        rows.append(("header", header, None, None, None))
        for sub, group in subgroups:
            if sub:
                rows.append(("sub", sub, None, None, None))
            part = bench[bench["group"] == group].sort_values("macro_auroc", ascending=False)
            for _, r in part.iterrows():
                label = SHORT_NAMES.get(r["model"], r["model"]) + (" †" if r["same_protocol"] != "yes" else "")
                rows.append(("pub", label, float(r["macro_auroc"]), None, r["same_protocol"]))
    rows.append(("header", f"This work (same test fold, {main['n']:,} records)", None, None, None))
    rows.append(("main", "General-purpose LLM with a Python sandbox", main["macro_auroc"],
                 main["macro_auroc_ci95"], "yes"))
    rows.append(("note", "input: raw signal + age, sex", None, None, None))
    rows.append(("ctrl", "Age-and-sex control (logistic regression)", ctrl["macro_auroc"],
                 ctrl["macro_auroc_ci95"], "yes"))
    colours = {"main": CODE, "ctrl": MUTED}
    text_colours = colours

    n = len(rows)
    pitch = 3.2 * MM
    top_pad, bottom_pad = 3 * MM, 9 * MM
    fig = plt.figure(figsize=(DOUBLE, n * pitch + top_pad + bottom_pad))
    H = fig.get_figheight()
    x_labels_end, ax_w = 0.37, 0.22
    ax = fig.add_axes([x_labels_end + 0.01, bottom_pad / H, ax_w, 1 - (top_pad + bottom_pad) / H])
    y = np.arange(n)[::-1]
    lo, hi = main["macro_auroc_ci95"]
    ax.axvspan(lo, hi, color=CODE, alpha=0.15, lw=0)
    ax.axvline(0.5, color=LIGHT, lw=0.6, ls=":")
    marker = {"yes": dict(marker="o", mfc=INK, mec=INK), "unclear": dict(marker="o", mfc="white", mec=INK),
              "no": dict(marker="D", mfc="white", mec=INK)}
    in_axes = transforms.blended_transform_factory(ax.transAxes, ax.transData)
    in_fig = transforms.blended_transform_factory(fig.transFigure, ax.transData)
    for yi, (kind, label, v, ci, proto) in zip(y, rows):
        if kind == "header":
            ax.text(0.012, yi, label, transform=in_fig, ha="left", va="center", fontweight="bold")
            continue
        if kind == "sub":  # a white box where a long heading crosses the chance line
            ax.text(0.022, yi, label, transform=in_fig, ha="left", va="center", style="italic", color=MUTED,
                    bbox=dict(fc="white", ec="none", pad=0.5))
            continue
        if kind == "note":
            ax.text(x_labels_end, yi, label, transform=in_fig, ha="right", va="center", color=CODE)
            continue
        colour = colours.get(kind, INK)
        if ci is not None:
            ax.plot(ci, [yi, yi], color=colour, lw=1.2, solid_capstyle="butt")
            ax.plot([v], [yi], "o", ms=4.6, color=colour, mec="white", mew=0.5)
        else:
            ax.plot([v], [yi], ls="none", ms=3.4, mew=0.7, **marker[proto])
        tcol = text_colours.get(kind, INK)
        ax.text(x_labels_end, yi, label, transform=in_fig, ha="right", va="center",
                color=tcol if kind != "pub" else INK, fontweight="bold" if kind == "main" else "normal")
        ax.text(1.02, yi, f"{v:.3f}", transform=in_axes, ha="left", va="center",
                color=tcol if kind != "pub" else MUTED, fontweight="bold" if kind == "main" else "normal")
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0.48, 0.96)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9])
    ax.set_xlabel("Macro AUROC (5 superclasses)", labelpad=2)
    panel_label(ax, "a", x=-x_labels_end / ax_w - 0.02 + 0.012 / ax_w + 0.04, y=1.0)

    # b: per class
    axb_h = 58 * MM / H
    axb = fig.add_axes([0.715, 1 - top_pad / H - axb_h - 5 * MM / H, 0.265, axb_h])
    xs = np.arange(len(LABELS))
    for k, (summ, colour, dx) in enumerate([(main, CODE, -0.12), (ctrl, MUTED, 0.12)]):
        vals = [summ["per_class_auroc"][l] for l in LABELS]
        cis = summ.get("per_class_ci95")
        if cis:
            for x, l in zip(xs, LABELS):
                axb.plot([x + dx, x + dx], cis[l], color=colour, lw=1.0, solid_capstyle="butt")
        axb.plot(xs + dx, vals, "o", ms=3.8, color=colour, mec="white", mew=0.4, ls="none")
    axb.set_xticks(xs)
    axb.set_xticklabels(LABELS)
    axb.set_ylim(0.6, 1.0)
    axb.set_ylabel("Test AUROC (95% CI)", labelpad=2)
    panel_label(axb, "b", x=-0.2, y=1.0)

    other = {int(mm.replace(",", "")) for src in bench.loc[bench["same_protocol"] == "no", "source"]
             for mm in re.findall(r"(\d{1,3}(?:,\d{3})+|\d{3,5})[- ]record", str(src))}
    other_txt = f"\n({' or '.join(f'{k:,}' for k in sorted(other))} records)" if other else ""
    legend_kw = dict(loc="upper left", ncol=1, handletextpad=0.5, borderaxespad=0.0, handlelength=1.5,
                     labelspacing=0.45, title_fontproperties={"weight": "bold", "size": 8}, alignment="left")
    this_work = [Line2D([], [], color=c, lw=1.2, marker="o", ms=4.6, mec="white", mew=0.5, label=lab)
                 for c, lab in [(CODE, "General-purpose LLM with a Python sandbox"), (MUTED, "Age-and-sex control")]]
    fig.legend(handles=this_work, bbox_to_anchor=(0.65, axb.get_position().y0 - 9 * MM / H),
               title="This work (a, b): estimate and 95% CI", **legend_kw)
    handles = [
        Line2D([], [], ls="none", ms=3.4, mew=0.7, **marker["yes"], label="Published, benchmark test fold (fold 10)"),
        Line2D([], [], ls="none", ms=3.4, mew=0.7, **marker["unclear"], label="† Published, split not stated"),
        Line2D([], [], ls="none", ms=3.4, mew=0.7, **marker["no"], label=f"† Published, other test set{other_txt}"),
        plt.Rectangle((0, 0), 1, 1, color=CODE, alpha=0.15, lw=0, label="95% CI of the model"),
        Line2D([], [], color=LIGHT, lw=0.6, ls=":", label="Chance (0.5)"),
    ]
    fig.legend(handles=handles, bbox_to_anchor=(0.65, axb.get_position().y0 - 27 * MM / H), title="Panel a",
               **legend_kw)
    save(fig, "fig2_benchmark.pdf")


# --- Figure 3: by statement, and stated measurements -----------------------------------------------

# (reference, measure, statistic, label): the row of posthoc_results.json "measurements" with this reference
# and measure whose statistic matches the regex (None: the only such row); in the label, {stat} is that statistic
# and {n} the number of records with the statement
MEAS_ROWS = [
    ("CLBBB", "qrs", None, "CLBBB: {stat} ({n})"),
    ("CRBBB", "qrs", None, "CRBBB: {stat} ({n})"),
    ("IRBBB", "qrs", None, "IRBBB: {stat} ({n})"),
    ("IVCD", "qrs", None, "IVCD: {stat} ({n})"),
    ("1AVB", "pr", None, "1AVB: {stat} ({n})"),
    ("LAFB", "axis", r"-\s*45", "LAFB: {stat} ({n})"),  # the AHA/ACCF/HRS criterion; the -30° row is in the measurement table
    ("SBRAD", "hr", None, "SBRAD: {stat} ({n})"),
    ("STACH", "hr", None, "STACH: {stat} ({n})"),
    ("LVH", "sokolow", None, "LVH: {stat} ({n})"),
    ("AFIB or AFLT", "af", None, "AFIB/AFLT: AF named ({n})"),
    # here consistency means a short bar: in sinus rhythm the key findings should not name AF
    ("SR, no AFIB or AFLT", "af", None, "Sinus rhythm: AF named ({n}),\nshould be absent"),
]


def share_text(x):
    """A share as the text gives it (analysis/posthoc.py macros): whole percent, one decimal below 5%."""
    return f"{100 * x:.0f}%" if x >= 0.05 else f"{100 * x:.1f}%"


def meas_rows(measurements):
    """(row, label) for MEAS_ROWS: the one row of the list with that reference and measure whose statistic matches
    (selected from the list itself, so that duplicate rows fail the check as well)."""
    out = []
    for ref, w, stat, lab in MEAS_ROWS:
        hits = [r for r in measurements
                if (r["reference"], r["measure"]) == (ref, w) and (stat is None or re.search(stat, r["statistic"]))]
        if len(hits) != 1:
            raise ValueError(f"{len(hits)} measurement rows for {ref}, {w}, statistic {stat}")
        out.append((hits[0], lab.format(stat=tex_text(hits[0]["statistic"]), n=f"{hits[0]['n']:,}")))
    return out


def fig3_where():
    ph = json.loads((ROOT / "analysis" / "posthoc_results.json").read_text())
    stm = ph["statements"]
    fig = plt.figure(figsize=(DOUBLE, 112 * MM))
    # a: AUROC by statement
    ax = fig.add_axes([0.165, 0.09, 0.26, 0.80])
    order = sorted(stm, key=lambda r: (LABELS.index(r["class"]), -r["n"]))
    y, yy, last = [], 0.0, None
    for r in order:
        if last is not None and r["class"] != last:
            yy -= 0.6
        y.append(yy)
        yy -= 1
        last = r["class"]
    trans = transforms.blended_transform_factory(ax.transAxes, ax.transData)
    for yi, r in zip(y, order):
        lo, hi = r["ci"]
        col = CLASS_COLOURS[r["class"]]
        ax.plot([lo, hi], [yi, yi], color=col, lw=1.0, solid_capstyle="butt")
        ax.plot([r["code"]], [yi], "o", ms=3.8, color=col, mec="white", mew=0.4, clip_on=False)  # whole at 1.0
        ax.text(-0.03, yi, f"{r['statement']} ({r['n']})", transform=trans, ha="right", va="center")
        ax.text(1.03, yi, f"{r['code']:.3f}", transform=trans, ha="left", va="center", color=MUTED)
    ax.text(-0.03, y[0] + 1.9, "Statement (records)", transform=trans, ha="right", fontsize=7.5, va="center",
            fontweight="bold")
    for c in LABELS:
        ys = [yi for yi, r in zip(y, order) if r["class"] == c]
        if ys:
            ax.text(-0.58, np.mean(ys), c, transform=trans, ha="center", va="center", color=CLASS_COLOURS[c],
                    fontweight="bold", rotation=90)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_ylim(min(y) - 0.7, y[0] + 2.4)
    ax.set_xlim(0.6, 1.0)
    ax.set_xticks([0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_xlabel("AUROC (95% CI)", labelpad=2)
    panel_label(ax, "a", x=-0.52, y=1.12)

    # b: measurements; at right, the share meeting the criterion among the records that stated a value (the "Met"
    # column of the measurement table (\MeasurementTable), as in the text)
    rows = meas_rows(ph["measurements"])
    left_b, w_b = 0.765, 0.165
    axb = fig.add_axes([left_b, 0.09, w_b, 0.80])
    in_b = transforms.blended_transform_factory(axb.transAxes, axb.transData)
    fx = lambda x: (x - left_b) / w_b  # noqa: E731   # figure fraction as axes fraction
    yb = np.arange(len(rows))[::-1].astype(float)
    h = 0.6
    for yi, (r, lab) in zip(yb, rows):
        crit = r["code_stat"]
        stated = r["code_stated"] if r["measure"] != "af" else 1.0
        # share of all records with the statement that state a value meeting the criterion
        share = (crit * stated) if np.isfinite(crit) else 0.0
        if r["measure"] != "af":
            axb.barh(yi, stated, height=h, color=CODE, alpha=0.25, lw=0)
        axb.barh(yi, share, height=h, color=CODE, lw=0)
        axb.text(fx(left_b - 0.007), yi, lab, transform=in_b, ha="right", va="center", linespacing=1.15)
        axb.text(fx(0.98), yi, share_text(crit) if np.isfinite(crit) else "–", transform=in_b, ha="right",
                 va="center", color=MUTED)
    axb.text(fx(0.98), yb[0] + 0.85, "Met", transform=in_b, ha="right", va="center", fontsize=7.5,
             fontweight="bold")
    axb.set_yticks([])
    axb.spines["left"].set_visible(False)
    axb.set_xlim(0, 1)
    axb.set_xticks([0, 0.5, 1])
    axb.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axb.set_xlabel("Share of records with the statement", labelpad=2)
    axb.set_ylim(-0.7, len(rows) - 0.1)
    handles = [plt.Rectangle((0, 0), 1, 1, color=CODE, label="Value stated and meets criterion"),
               plt.Rectangle((0, 0), 1, 1, color=CODE, alpha=0.25, lw=0, label="Value stated")]
    axb.legend(handles=handles, loc="lower left", bbox_to_anchor=(fx(0.507), 1.01), ncol=2, handlelength=1.0,
               handletextpad=0.4, columnspacing=0.8, borderaxespad=0.2)
    panel_label(axb, "b", x=fx(0.503), y=1.12)
    save(fig, "fig3_where.pdf")


# --- Supplementary figures --------------------------------------------------------------------------

def figS1_curves():
    m = meta()
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 66 * MM), squeeze=False)
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.17, top=0.93, wspace=0.34)
    for row, (run, name) in enumerate([(TEST, "")]):
        pred = preds(run)
        ids = sorted(pred.index)
        y = truth(m, ids)
        p = pred.loc[ids, PROB_COLS].to_numpy(float)
        ax_roc, ax_pr, ax_rel = axes[row]
        for j, label in enumerate(LABELS):
            c = CLASS_COLOURS[label]
            fpr, tpr, _ = roc_curve(y[:, j], p[:, j])
            ax_roc.plot(fpr, tpr, color=c, lw=1.0,
                        label=f"{label} {roc_auc_score(y[:, j], p[:, j]):.3f}")
            prec, rec, _ = precision_recall_curve(y[:, j], p[:, j])
            ax_pr.plot(rec, prec, color=c, lw=1.0, drawstyle="steps-post",
                       label=f"{label} {average_precision_score(y[:, j], p[:, j]):.3f}")
            idx = calib_bins(p[:, j])  # ten equal-width bins, as for the ECE (analysis/posthoc.py)
            xs, ys = [], []
            for b in range(10):
                mm = idx == b
                if mm.sum() >= 20:
                    xs.append(p[mm, j].mean())
                    ys.append(y[mm, j].mean())
            ax_rel.plot(xs, ys, color=c, lw=0.9, marker="o", ms=2.8, mec="white", mew=0.4, clip_on=False)
        ax_roc.plot([0, 1], [0, 1], color=LIGHT, lw=0.6, ls="--")
        ax_rel.plot([0, 1], [0, 1], color=LIGHT, lw=0.6, ls="--")
        ax_roc.set(xlabel="False positive rate", ylabel="True positive rate")
        ax_pr.set(xlabel="Recall", ylabel="Precision")
        ax_rel.set(xlabel="Predicted probability (bin mean)", ylabel="Observed frequency")
        ax_roc.legend(loc="lower right", title="AUROC", fontsize=7, title_fontsize=7, handlelength=1.2)
        ax_pr.legend(loc="lower left", title="AUPRC", fontsize=7, title_fontsize=7, handlelength=1.2)
        for ax in (ax_roc, ax_pr, ax_rel):
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.0)
            ax.set_xticks([0, 0.5, 1])
            ax.set_yticks([0, 0.5, 1])
            ax.set_aspect("equal")
        for ax, letter in zip(axes[row], "abc" if row == 0 else "def"):
            panel_label(ax, letter, x=-0.22, y=1.0)
    save(fig, "figS1_curves.pdf")


def figS2_usage():
    lines = [r for r in results_lines(TEST) if r.get("type") == "succeeded"]
    runs_ = np.array([r["code_runs"] for r in lines])
    usd = np.array([r["usd"] for r in lines])
    out = np.array([r["usage"]["output_tokens"] for r in lines])
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 52 * MM))
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.22, top=0.92, wspace=0.35)
    k = np.arange(runs_.min(), runs_.max() + 1)
    axes[0].bar(k, [(runs_ == v).sum() for v in k], color=CODE, width=0.7)
    axes[0].set(xlabel="Code runs per record", ylabel="Records")
    axes[1].hist(out, bins=40, color=CODE)
    axes[1].set(xlabel="Output tokens per record", ylabel="Records")
    axes[2].hist(usd, bins=40, color=CODE)
    axes[2].axvline(usd.mean(), color=INK, lw=0.8, ls="--")
    axes[2].text(usd.mean(), axes[2].get_ylim()[1] * 0.95, f" mean USD {usd.mean():.3f}", ha="left", va="top")
    axes[2].set(xlabel="Cost per record (USD, batch price)", ylabel="Records")
    for ax, letter in zip(axes, "abc"):
        panel_label(ax, letter, x=-0.2, y=1.0)
    save(fig, "figS2_usage.pdf")


FIGURES = {"fig1": fig1_example, "fig2": fig2_benchmark, "fig3": fig3_where, "figS1": figS1_curves,
           "figS2": figS2_usage}

if __name__ == "__main__":
    wanted = sys.argv[1:] or list(FIGURES)
    for k in wanted:
        print(f"[{k}]")
        out = FIGURES[k]()
        if out:
            print(out)
