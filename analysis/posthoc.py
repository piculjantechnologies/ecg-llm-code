"""Post hoc and exploratory analyses of the test run (not pre-specified; written after the test run).

Run from the repository root:  python analysis/posthoc.py
Writes paper/generated/posthoc.tex (macros prefixed PH, plus \\CharacteristicsTable, \\SubgroupTable,
\\StatementTable, \\MeasurementTable and \\ProvenanceTable) and analysis/posthoc_results.json. No API
calls; PTB-XL only.

A. characteristics of the benchmark splits
B. subgroups (sex, age, rhythm, signal-quality flags, extra beats, device): macro AUROC
C. diagnostic statements: AUROC of the superclass probability for records with one statement against
   records without that superclass
D. measurements stated in the key findings (heart rate, PR, QRS, QTc, axis, Sokolow-Lyon voltage,
   atrial fibrillation) by PTB-XL statement or metadata field
E. what the model did in the sandbox (code runs, errors, methods used, what the code printed)
F. calibration
G. MI discrimination by the likelihood the annotator gave the MI statement
H. patient-level bootstrap of the primary endpoint
"""
import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from common import (AGESEX_TEST, GEN, LABELS, OUT_JSON, PROB_COLS, TEST, TEST_FOLD, TRAIN_FOLDS, VAL_FOLD, auc,
                    benchmark_split, boots_for, ci, custom_to_anon, f3, key, meta, preds, results_lines, summary,
                    thousands, write_json, write_macros)
from ecgcode.evaluate import truth

RNG_SEED = 0
MIN_SUBGROUP = 100     # smallest subgroup reported
MIN_POSITIVES = 10     # a subgroup with fewer positives in a class is flagged
MIN_STATEMENT = 20     # smallest number of test records with a statement for the statement table
N_STMT_BOOT = 2000     # stratified bootstrap resamples per statement


def pct(x, d=0):
    return f"{100 * x:.{d}f}"


def tex_escape(s):
    return s.replace("_", "\\_").replace("&", "\\&").replace("%", "\\%")


# --- A. characteristics --------------------------------------------------------------------------

NOISE_COLS = ["baseline_drift", "static_noise", "burst_noise", "electrodes_problems"]


def has_code(codes, c):
    return c in codes


def characteristics(m):
    rows = []
    for name, folds in [("Training (folds 1--8)", TRAIN_FOLDS), ("Validation (fold 9)", VAL_FOLD),
                        ("Test (fold 10)", TEST_FOLD)]:
        s = benchmark_split(m, folds)
        age = s["age"].where(s["age"] < 300)
        row = {"split": name, "records": len(s), "patients": s["patient_id"].nunique(),
               "age_median": float(age.median()), "age_q1": float(age.quantile(.25)), "age_q3": float(age.quantile(.75)),
               "female": float((s["sex"] == 1).mean()),
               "afib": float(s["scp_codes"].apply(lambda d: "AFIB" in d or "AFLT" in d).mean()),
               "noise": float(s[NOISE_COLS].notna().any(axis=1).mean())}
        for l in LABELS:
            row[l] = float(s[f"y_{l}"].mean())
        rows.append(row)
    lines = ["\\newcommand{\\CharacteristicsTable}{%", "\\begin{table}[!htbp]\\centering",
             "\\caption{\\textbf{Characteristics of the benchmark splits.} Records with at least one superclass "
             "label, in the three sets of the benchmark split. In the ``Age'' row, ages above 89, which PTB-XL codes as 300, "
             "are excluded from the median. The ``Signal-quality flag'' row gives the share of records with baseline "
             "drift, static noise, burst noise or electrode problems noted in the PTB-XL metadata.}",
             "\\label{tab:characteristics}\\small",
             "\\begin{tabular}{lrrr}\\toprule", " & " + " & ".join(r["split"] for r in rows) + " \\\\ \\midrule",
             "Records & " + " & ".join(thousands(r["records"]) for r in rows) + " \\\\",
             "Patients & " + " & ".join(thousands(r["patients"]) for r in rows) + " \\\\",
             "Age, median (IQR), years & " + " & ".join(f"{r['age_median']:.0f} ({r['age_q1']:.0f}--{r['age_q3']:.0f})"
                                                        for r in rows) + " \\\\",
             "Female, \\% & " + " & ".join(pct(r["female"], 1) for r in rows) + " \\\\"]
    for l in LABELS:
        lines.append(f"{l}, \\% & " + " & ".join(pct(r[l], 1) for r in rows) + " \\\\")
    lines += ["Atrial fibrillation or flutter, \\% & " + " & ".join(pct(r["afib"], 1) for r in rows) + " \\\\",
              "Signal-quality flag, \\% & " + " & ".join(pct(r["noise"], 1) for r in rows) + " \\\\",
              "\\bottomrule\\end{tabular}\\end{table}}"]
    return rows, "\n".join(lines)


# --- B. subgroups ------------------------------------------------------------------------------

def subgroups(m, code, ids):
    t = m.loc[ids]
    age = t["age"].where(t["age"] < 300, 90)
    noise = t[NOISE_COLS].notna().any(axis=1)
    af = t["scp_codes"].apply(lambda d: "AFIB" in d or "AFLT" in d)
    groups = [("Sex", "Female", t["sex"] == 1), ("Sex", "Male", t["sex"] == 0),
              ("Age", "Under 50 years", age < 50), ("Age", "50--69 years", (age >= 50) & (age < 70)),
              ("Age", "70 years or older", age >= 70),
              ("Rhythm", "Atrial fibrillation or flutter", af), ("Rhythm", "Other rhythms", ~af),
              ("Signal quality", "Flag noted", noise), ("Signal quality", "No flag", ~noise),
              ("Extra beats", "Noted", t["extra_beats"].notna()), ("Extra beats", "Not noted", t["extra_beats"].isna())]
    dev = t["device"].str.strip()
    for d, n in dev.value_counts().items():
        if n >= MIN_SUBGROUP:
            groups.append(("Device", tex_escape(re.sub(r"\s+", " ", d)), dev == d))
    out = []
    for kind, name, mask in groups:
        sel = [i for i, keep in zip(ids, mask.to_numpy()) if keep]
        row = {"kind": kind, "group": name, "n": len(sel),
               "positives": dict(zip(LABELS, truth(m, sel).sum(axis=0).astype(int).tolist()))}
        try:
            b = boots_for(sel)
        except ValueError as e:
            row["note"] = str(e)
            out.append(row)
            continue
        sc = summary(code, sel, b)
        row.update({"code": sc["macro_auroc"], "code_ci": sc["macro_auroc_ci95"], "per_class": sc["per_class_auroc"]})
        out.append(row)
    n_dev = sum(r["n"] for r in out if r["kind"] == "Device")
    lines = ["\\newcommand{\\SubgroupTable}{%", "\\begin{table}[!htbp]\\centering",
             "\\caption{\\textbf{Macro AUROC by subgroup, and MI discrimination by the annotator's likelihood, on the test "
             "fold (post hoc).} 95\\% percentile bootstrap "
             f"CIs ({thousands(len(boots_for(ids)))} resamples). In the upper part, the ``Records'' column gives the number "
             "of test records in the subgroup and the ``Positives'' column the number of them with each superclass "
             f"label (NORM/MI/STTC/CD/HYP). $^\\dagger$Fewer than {MIN_POSITIVES} positives in a class, so the AUROC of that "
             "class, which counts one fifth of the macro AUROC, rests on few records. The device rows show the PTB-XL "
             f"device values with at least {MIN_SUBGROUP} test records ({thousands(n_dev)} of {thousands(len(ids))} records), "
             "as recorded; values that may denote one device model (for example AT-6~C 5.5, AT-6~C and AT-6~C 5.8; "
             "PTB-XL does not document the suffixes) are kept separate, and devices are confounded with recording "
             "sites. Signal-quality flag and extra "
             "beats as noted in the PTB-XL metadata. In the lower part, MI-positive records are grouped by the highest "
             "likelihood (0--100) that the annotator assigned among their MI statements; the ``MI-positive records'' "
             "column gives their number, the ``MI AUROC'' column the AUROC of the MI probability for each group "
             "against all test records without MI, with 95\\% CIs from "
             f"{thousands(N_STMT_BOOT)} stratified bootstrap resamples (positives and negatives resampled "
             "separately), and the ``Age-and-sex control'' column the AUROC of the control on the same records.}",
             "\\label{tab:subgroups}\\footnotesize\\setlength{\\tabcolsep}{4pt}",
             "\\begin{tabular}{llrcc}\\toprule",
             " & Subgroup & Records & Positives & Macro AUROC [95\\% CI] \\\\ \\midrule"]
    last = None
    for r in out:
        kind = r["kind"] if r["kind"] != last else ""
        last = r["kind"]
        few = "$^\\dagger$" if min(r["positives"].values()) < MIN_POSITIVES else ""
        pos = "/".join(str(r["positives"][l]) for l in LABELS)
        if "code" not in r:
            lines.append(f"{kind} & {r['group']}{few} & {thousands(r['n'])} & {pos} & not estimable \\\\")
            continue
        lines.append(f"{kind} & {r['group']}{few} & {thousands(r['n'])} & {pos} & {f3(r['code'])} "
                     f"[{f3(r['code_ci'][0])}, {f3(r['code_ci'][1])}] \\\\")
    lines += ["\\bottomrule\\end{tabular}\\end{table}}"]
    return out, "\n".join(lines)


# --- C. statements -----------------------------------------------------------------------------

def statement_table(m, code, ids):
    st = pd.read_csv(m.attrs["root"] + "/scp_statements.csv", index_col=0)
    diag = st[st["diagnostic"] == 1]
    t = m.loc[ids]
    counts = Counter(c for codes in t["scp_codes"] for c in codes if c in diag.index)
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for code_name, n in counts.most_common():
        if n < MIN_STATEMENT:
            continue
        cls = diag.loc[code_name, "diagnostic_class"]
        pos = t["scp_codes"].apply(lambda d, c=code_name: c in d).to_numpy()
        neg = (t[f"y_{cls}"] == 0).to_numpy()
        pc = code.loc[ids, f"p_{cls}"].to_numpy()
        ip, ineg = np.where(pos)[0], np.where(neg)[0]
        y = np.r_[np.ones(len(ip)), np.zeros(len(ineg))]
        a_c = auc(y, np.r_[pc[ip], pc[ineg]])
        boots = []
        for _ in range(N_STMT_BOOT):   # stratified bootstrap: positives and negatives resampled separately
            bp, bn = rng.choice(ip, len(ip)), rng.choice(ineg, len(ineg))
            boots.append(auc(y, np.r_[pc[bp], pc[bn]]))
        rows.append({"statement": code_name, "description": diag.loc[code_name, "description"], "class": cls,
                     "n": int(n), "code": a_c, "ci": ci(boots)})
    rows.sort(key=lambda r: (LABELS.index(r["class"]), -r["n"]))
    lines = ["\\newcommand{\\StatementTable}{%", "\\begin{table}[!htbp]\\centering",
             "\\caption{\\textbf{Discrimination by diagnostic statement on the test fold (post hoc, exploratory).} "
             f"Statements with at least {MIN_STATEMENT} test records, with codes and descriptions as in PTB-XL; the "
             "``Class'' column gives the superclass of each statement. The ``Records'' column gives the number of "
             "test records with the statement (any likelihood), and the ``AUROC'' column the AUROC of the superclass "
             "probability for these records against all test records without that superclass, with 95\\% CIs from "
             f"{thousands(N_STMT_BOOT)} stratified bootstrap resamples.}}",
             "\\label{tab:statements}\\footnotesize\\setlength{\\tabcolsep}{4pt}",
             # descriptions ragged right and not hyphenated
             "\\begin{tabular}{ll>{\\raggedright\\arraybackslash\\hyphenpenalty=10000\\exhyphenpenalty=10000}"
             "p{6.6cm}rc}\\toprule",
             "Class & Code & Statement & Records & AUROC [95\\% CI] \\\\ \\midrule"]
    last = None
    for r in rows:
        cls = r["class"] if r["class"] != last else ""
        if last is not None and r["class"] != last:
            lines.append("\\addlinespace")
        last = r["class"]
        lines.append(f"{cls} & {tex_escape(r['statement'])} & {tex_escape(r['description'])} & {r['n']} & "
                     f"{f3(r['code'])} [{f3(r['ci'][0])}, {f3(r['ci'][1])}] \\\\")
    lines += ["\\bottomrule\\end{tabular}\\end{table}}"]
    return rows, "\n".join(lines)


# --- D. stated measurements ----------------------------------------------------------------------

def _norm(text):
    # the model writes en and em dashes only between numbers or leads (ranges), so they become ' to '
    return (str(text).replace("−", "-").replace("–", " to ").replace("—", " to ")
            .replace("~", " about ").replace("≈", " about "))


# A stated value is a number followed by the measurement's unit, or a range ('a-b', 'a to b', 'between a and b')
# whose midpoint is used. A '-' right after a number separates a range; it is not a sign ('+105-110°'); a '+' before
# the unit is ignored ('500+ ms').
MEASURES = ["hr", "pr", "qrs", "qtc", "axis", "sokolow"]
NUMBER = r"(?:(?<![\d.])(?<![\d.]\s)[+-])?(?<![\d.])\d+(?:\.\d+)?"
UNITS = {"hr": r"(?:bpm|beats per minute|beats/min|/min|per minute)", "pr": r"ms", "qrs": r"ms", "qtc": r"ms",
         "axis": r"(?:°|degrees?\b)", "sokolow": r"mV"}
ANCHOR = {  # the label that opens the clause holding the value
    "hr": re.compile(r"\b(?:ventricular rate|ventricles|heart rate|HR|rate|rhythm|tachycardia|bradycardia)\b", re.I),
    # or a P wave placed before the QRS: 'a P wave sits about 320 ms before every QRS'
    "pr": re.compile(r"\bPR\b(?!-segment| segment)"
                     r"|\bP(?:-like)?(?:[- ]wave)?\b(?=[^.;]{0,40}?\bms before (?:each|every|most) QRS)"),
    "qrs": re.compile(r"QRS\b(?! axis| component)"),
    "qtc": re.compile(r"QTc\b"),
    "axis": re.compile(r"\baxis\b", re.I),
    # named, or an unnamed sum 'S V1 (x) + R V5 (y) = z mV'
    "sokolow": re.compile(r"Sokolow\b|\bS\s?V1\b(?:\s*(?:of\s+)?(?:about\s+)?[\d.]+\s*(?:mV)?)?\s*(?:\+|plus)\s*"
                          r"R\s?V[56]", re.I),
}
STOP = {  # another measurement's label within the clause: a value after it belongs to that measurement
    "hr": re.compile(r"\bPR\b|QRS|QT|\baxis\b|\bRR\b|atrial activity|flutter waves|P waves? (?:rate|at)", re.I),
    "pr": re.compile(r"QRS|QT|\bRR\b|\baxis\b|bpm|/min|P peak"),
    "qrs": re.compile(r"\bPR\b|QT|\bRR\b|\baxis\b|bpm|/min|P[- ]wave"),
    "qtc": re.compile(r"QRS|\bPR\b|\bRR\b|\baxis\b|bpm|/min|\bQT\b"),   # 'QTc (QT 420 ms, QTc 485 ms)'
    "axis": re.compile(r"QRS|\bPR\b|QT|\bms\b|\bmV\b|bpm|/min"),
    "sokolow": re.compile(r"Cornell", re.I),
}
PLAUSIBLE = {"hr": (20, 250), "pr": (40, 450), "qrs": (20, 250), "qtc": (150, 700), "axis": (-180, 270),
             "sokolow": (0.3, 10)}   # both ends of a stated range must lie within


def _range(unit):
    """A single value or a range; a unit after the first number of a range is optional."""
    return (rf"(?:between\s+(?:about\s+)?(?P<b1>{NUMBER})\s*(?:{unit})?\s*and\s+(?:about\s+)?(?P<b2>{NUMBER})"
            rf"|(?P<r1>{NUMBER})\s*(?:{unit})?\s*(?:to|-)\s*(?:about\s+)?(?P<r2>{NUMBER})"
            rf"|(?P<v>{NUMBER}))")


def _value_rx(unit):
    return re.compile(_range(unit) + rf"\s*\+?\s*{unit}")


VALUE = {w: _value_rx(u) for w, u in UNITS.items()}
# an axis written without its unit right after its label and before ',', ';' or the sentence end:
# 'the axis is about +90 to +100, QRS voltage is low' (used only when no axis with a unit is found)
AXIS_NO_UNIT = re.compile(r"\baxis\s+(?:(?:is|of)\s+)?(?:(?:about|approximately|around|roughly)\s+)?"
                          + _range(UNITS["axis"]) + r"(?=\s*(?:[,;]|\.(?:\s|$)|$))", re.I)
PT_AXIS = re.compile(r"\b[PT](?:[- ]?waves?)?[- ]$")   # 'P axis', 'T-wave axis': not the QRS axis
# a Sokolow-Lyon value computed with S in V2 instead of S in V1 ('3.2 mV by S V2 + R V5, 2.2 mV strict'): a later
# value in the same clause is taken instead, if there is one
S_V2 = re.compile(r"\s*(?:by|using|with)\s+(?:the\s+)?S\s*(?:in\s+)?V2\b")
BEFORE_LABEL = {  # a value written before its label: '61 bpm sinus rhythm', '+80° axis', 'a narrow 76 ms QRS'
    "hr": VALUE["hr"], "axis": re.compile(VALUE["axis"].pattern + r"\s*(?:\)|,)?\s*(?:frontal\s+)?axis"),
    "qrs": re.compile(VALUE["qrs"].pattern + r"\s*QRS")}
SPECIAL = {"qtc": re.compile(r"QT/QTc\D{0,12}?\d{3}\s*/\s*(\d{3})\s*ms"),   # 'QT/QTc about 350/405 ms'
           "sokolow": re.compile(rf"Sokolow(?:-Lyon)?\s+(?:about\s+)?({NUMBER}),\s*Cornell\s+(?:about\s+)?[\d.]+\s*mV")}
HR_NO_UNIT = re.compile(r"\b(?:HR|heart rate|rate)\b\D{0,14}?(\d{2,3})\b(?!\s*(?:ms|mV|°|%|s\b))")   # 'mean HR about 94'
SUMMAND = re.compile(r"(?<![A-Za-z\d.])\d+(?:\.\d+)?\s*(?:mV)?\s*(?:\+|plus)\s*R\s?V[56](?:/V?6)?\s*(?:of|is|:)?\s*"
                     r"(?:about)?$")


def _clause_end(s, start):
    """';', a sentence end, or a ')' that closes a parenthesis opened before start."""
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "(":
            depth += 1
        elif s[i] == ")":
            depth -= 1
            if depth < 0:
                return i
        elif s[i] == ";" or (s[i] == "." and (i + 1 == len(s) or s[i + 1].isspace())):
            return i
    return len(s)


def _ends(mt):
    """Both ends of a matched value (equal for a single value); '-30 to 45°' means -30 to -45."""
    a, b = (mt.group("b1"), mt.group("b2")) if mt.group("b1") else (mt.group("r1"), mt.group("r2"))
    if a is None:
        return float(mt.group("v")), float(mt.group("v"))
    if float(a) < 0 and not b.startswith(("+", "-")):
        b = "-" + b
    return float(a), float(b)


def _summand(s, mt):
    """A Sokolow-Lyon component, not the sum: followed by '+' or by '=' and a number, preceded by '+', or the R V5
    value in 'S V1 1.1 + R V5 2.8'. The sum in 'Sokolow-Lyon 3.65 mV = S V1 1.30 + R V5 2.35' is not a component."""
    after, before = s[mt.end():mt.end() + 20].lstrip(), s[max(0, mt.start() - 40):mt.start()].rstrip()
    return (after.startswith("+") or bool(re.match(r"=\s*(?:about\s+)?\d", after)) or before.endswith("+")
            or bool(SUMMAND.search(before)))


def _pt_axis(s, start):
    """The 'axis' label at start belongs to the P or T wave."""
    return bool(PT_AXIS.search(s[:start]))


def stated(text, what):
    """The value the key findings state for a measurement (midpoint of a stated range), or NaN: the first value in
    the clause after its label, up to ';', the sentence end, the closing parenthesis or another measurement's label."""
    s = _norm(text)
    lo, hi = PLAUSIBLE[what]
    ok = lambda a, b: lo <= a <= hi and lo <= b <= hi  # noqa: E731
    mt = SPECIAL[what].search(s) if what in SPECIAL else None
    if mt:
        return float(mt.group(1))
    for an in ANCHOR[what].finditer(s):
        if what == "axis" and _pt_axis(s, an.start()):
            continue
        end = _clause_end(s, an.end())
        stop = STOP[what].search(s, an.end(), end)
        variant = None   # a Sokolow-Lyon value computed with S in V2
        for mt in VALUE[what].finditer(s, an.end(), stop.start() if stop else end):
            a, b = _ends(mt)
            if (what == "sokolow" and _summand(s, mt)) or not ok(a, b):
                continue
            if what == "sokolow" and S_V2.match(s, mt.end()):
                variant = (a + b) / 2 if variant is None else variant
                continue
            return (a + b) / 2
        if variant is not None:
            return variant
    for mt in BEFORE_LABEL[what].finditer(s) if what in BEFORE_LABEL else []:
        a, b = _ends(mt)
        if ok(a, b):
            return (a + b) / 2
    for mt in AXIS_NO_UNIT.finditer(s) if what == "axis" else []:
        a, b = _ends(mt)
        if not _pt_axis(s, mt.start()) and ok(a, b):
            return (a + b) / 2
    mt = HR_NO_UNIT.search(s) if what == "hr" else None
    return float(mt.group(1)) if mt and lo <= float(mt.group(1)) <= hi else np.nan


AF_RX = re.compile(r"atrial fibrillation|atrial flutter|\bAF\b|\bAFib\b|\bflutter\b", re.I)
NEG_AF_RX = re.compile(r"(?:no|not|without|rather than|rule out|rules out)\s+(?:\w+\s+){0,3}"
                       r"(?:atrial fibrillation|AF\b|flutter)", re.I)


def mentions_af(text):
    s = str(text)
    return bool(AF_RX.search(s)) and not NEG_AF_RX.search(s)


def measurement_audit(m, code, ids):
    t = m.loc[ids]
    codes = t["scp_codes"]
    has = lambda c: codes.apply(lambda d: c in d).to_numpy()  # noqa: E731
    norm_only = (t["labels"].apply(lambda ls: ls == ["NORM"])).to_numpy()
    sr_only = (has("SR") & ~has("AFIB") & ~has("AFLT"))
    af = has("AFIB") | has("AFLT")
    ha = t["heart_axis"].fillna("")
    kf = {"code": code.loc[ids, "key_findings"].fillna("")}
    vals = {k: {w: v.apply(lambda s, w=w: stated(s, w)).to_numpy() for w in MEASURES} for k, v in kf.items()}
    af_m = {k: v.apply(mentions_af).to_numpy() for k, v in kf.items()}
    rows = []

    def add(label, sel, what, stat_name, stat):
        r = {"reference": label, "n": int(sel.sum()), "measure": what, "statistic": stat_name}
        for k in ("code",):
            v = vals[k][what][sel] if what != "af" else None
            if what == "af":
                r[f"{k}_stated"] = 1.0
                r[f"{k}_median"] = np.nan
                r[f"{k}_stat"] = float(af_m[k][sel].mean())
                r[f"{k}_n"] = int(sel.sum())
                continue
            ok = v[np.isfinite(v)]
            r[f"{k}_stated"] = float(np.isfinite(v).mean()) if len(v) else np.nan
            r[f"{k}_n"] = int(len(ok))
            r[f"{k}_median"] = float(np.median(ok)) if len(ok) else np.nan
            r[f"{k}_stat"] = float(stat(ok)) if len(ok) else np.nan
        rows.append(r)

    wide = lambda v: (v >= 120).mean()  # noqa: E731
    add("CLBBB", has("CLBBB"), "qrs", "QRS $\\geq$120 ms", wide)
    add("CRBBB", has("CRBBB"), "qrs", "QRS $\\geq$120 ms", wide)
    add("IVCD", has("IVCD"), "qrs", "QRS $\\geq$120 ms", wide)   # the prompt: >= 120 ms for nonspecific widening
    add("IRBBB", has("IRBBB"), "qrs", "QRS 110--119 ms", lambda v: ((v >= 110) & (v < 120)).mean())
    add("NORM only", norm_only, "qrs", "QRS $<$110 ms", lambda v: (v < 110).mean())
    add("1AVB", has("1AVB"), "pr", "PR $>$200 ms", lambda v: (v > 200).mean())
    add("NORM only", norm_only, "pr", "PR 120--200 ms", lambda v: ((v >= 120) & (v <= 200)).mean())
    add("SBRAD", has("SBRAD"), "hr", "HR $<$60/min", lambda v: (v < 60).mean())
    add("STACH", has("STACH"), "hr", "HR $>$100/min", lambda v: (v > 100).mean())
    add("SR, no SBRAD, STACH or SARRH", sr_only & ~has("SBRAD") & ~has("STACH") & ~has("SARRH"), "hr",
        "HR 60--100/min", lambda v: ((v >= 60) & (v <= 100)).mean())
    add("LNGQT", has("LNGQT"), "qtc", "QTc $>$450 ms", lambda v: (v > 450).mean())
    add("NORM only", norm_only, "qtc", "QTc $\\leq$450 ms", lambda v: (v <= 450).mean())
    # the prompt gives no angle for fascicular blocks: left axis deviation and the AHA/ACCF/HRS LAFB criterion
    add("LAFB", has("LAFB"), "axis", "axis $\\leq$$-30^\\circ$", lambda v: (v <= -30).mean())
    add("LAFB", has("LAFB"), "axis", "axis $\\leq$$-45^\\circ$", lambda v: (v <= -45).mean())
    add("LPFB", has("LPFB"), "axis", "axis $\\geq$$+90^\\circ$", lambda v: (v >= 90).mean())
    add("Axis field ALAD", (ha == "ALAD").to_numpy(), "axis", "axis $\\leq$$-30^\\circ$", lambda v: (v <= -30).mean())
    add("Axis field LAD", (ha == "LAD").to_numpy(), "axis", "axis $<$$0^\\circ$", lambda v: (v < 0).mean())
    add("Axis field MID", (ha == "MID").to_numpy(), "axis", "axis $-30^\\circ$ to $+90^\\circ$",
        lambda v: ((v >= -30) & (v <= 90)).mean())
    add("Axis field RAD or ARAD", ha.isin(["RAD", "ARAD"]).to_numpy(), "axis", "axis $>$$+90^\\circ$",
        lambda v: (v > 90).mean())
    add("LVH", has("LVH"), "sokolow", "Sokolow--Lyon $\\geq$3.5 mV", lambda v: (v >= 3.5).mean())
    add("NORM only", norm_only, "sokolow", "Sokolow--Lyon $<$3.5 mV", lambda v: (v < 3.5).mean())
    add("AFIB or AFLT", af, "af", "AF or flutter named", None)
    add("SR, no AFIB or AFLT", sr_only, "af", "AF or flutter named", None)

    units = {"qrs": "ms", "pr": "ms", "hr": "/min", "qtc": "ms", "axis": "$^\\circ$", "sokolow": "mV", "af": ""}
    names = {"qrs": "QRS", "pr": "PR", "hr": "Heart rate", "qtc": "QTc", "axis": "Axis", "sokolow": "Sokolow--Lyon",
             "af": "Rhythm"}

    def fmt_med(x, what):   # math mode: a true minus sign
        if not np.isfinite(x):
            return "--"
        return f"${x:.1f}$" if what == "sokolow" else f"${x:.0f}$"

    # the definitions, criteria, extraction rules and abbreviations go in a note under the table, as wide as the
    # table (box 0), so that the table fits its page
    note = ("The ``Records with'' column gives a PTB-XL statement (any likelihood) or a value of the heart-axis "
            "field of the PTB-XL metadata, which is filled for part of the records, and the ``Criterion'' column the "
            "threshold that the stated value is checked against: the frozen prompt's where it gives one (QRS, PR, "
            "Sokolow--Lyon), otherwise a conventional limit; the prompt gives no angle for fascicular blocks, "
            "so LAFB is shown against left axis deviation ($\\leq -30^\\circ$) and against the AHA/ACCF/HRS criterion "
            "($\\leq -45^\\circ$)~\\citep{surawicz2009recommendations}. Extraction with regular expressions: the "
            "first value with its unit after the measurement's label, within the label's clause; otherwise a value "
            "written before the label (QRS, axis), for heart rate the first rate in beats per minute anywhere in the "
            "text or a rate given without its unit, and for the axis a value given without its unit directly after "
            "the label. PR also from a P wave's stated distance before the QRS. A range gives its midpoint. "
            "Sokolow--Lyon: the stated sum, not its components, and the value with S in V1 when one with S in V2 is "
            "also given. P- and T-wave axes are ignored. QTc as stated by the model (the prompt asked for Bazett's "
            "correction). "
            + "Statements: CLBBB and CRBBB, complete left and right bundle branch block; IVCD, non-specific "
            "intraventricular conduction disturbance; IRBBB, incomplete right bundle branch block; 1AVB, first-degree "
            "AV block; SR, SBRAD, STACH and SARRH, sinus rhythm, bradycardia, tachycardia and arrhythmia; LNGQT, long "
            "QT interval; LAFB and LPFB, left anterior and posterior fascicular block; LVH, left ventricular "
            "hypertrophy; AFIB and AFLT, atrial fibrillation and flutter. Heart-axis field: MID, normal; LAD, left "
            "axis deviation; ALAD, abnormal left axis deviation; RAD and ARAD, right and abnormal right axis "
            "deviation. ``AF or flutter named'': atrial fibrillation or flutter named, and not negated, in the key "
            "findings.")
    lines = ["\\newcommand{\\MeasurementTable}{%", "\\begin{table}[!htbp]\\centering",
             "\\caption{\\textbf{Measurements stated in the key findings, by PTB-XL statement or metadata field "
             "(test fold, post hoc).} The ``$n$'' column gives the number of test records; the ``Stated'' column, the "
             "share of them whose key findings state the measurement; the ``Median'' column, the median stated value; "
             "and the ``Met'' column, the share whose stated value meets the criterion. Median and Met are among the "
             "records that state the measurement (-- when fewer than 10 do). ``NORM only'': records whose only "
             "superclass is NORM. The statements are not measurements, so the table checks consistency, not "
             "accuracy.}",
             "\\label{tab:measurements}\\footnotesize\\setlength{\\tabcolsep}{4pt}\\renewcommand{\\arraystretch}{0.95}",
             "\\sbox0{\\begin{tabular}{lL{4.2cm}rrrr}\\toprule",
             "Records with & Criterion & $n$ & Stated & Median & Met \\\\ \\midrule"]
    last = None
    for r in rows:
        w = r["measure"]
        if w != last:
            if last is not None:
                lines.append("\\addlinespace[3pt]")
            lines.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{names[w]}" + (f" ({units[w]})" if units[w] else "")
                         + "}} \\\\")
        last = w
        crit = r["statistic"]

        def cells(k):
            if w == "af":
                return f"-- & -- & {pct(r[f'{k}_stat'], 1 if r[f'{k}_stat'] < 0.05 else 0)}\\%"
            st = pct(r[f"{k}_stated"]) + "\\%"
            if r[f"{k}_n"] < 10:   # too few stated values for a median or a share
                return f"{st} & -- & --"
            stat = (pct(r[f"{k}_stat"], 1 if r[f"{k}_stat"] < 0.05 else 0) + "\\%") if np.isfinite(r[f"{k}_stat"]) else "--"
            return f"{st} & {fmt_med(r[f'{k}_median'], w)} & {stat}"
        lines.append(f"\\quad {tex_escape(r['reference'])} & {crit} & {thousands(r['n'])} & {cells('code')} \\\\")
    lines += ["\\bottomrule\\end{tabular}}\\usebox0\\par\\smallskip",
              "\\begin{minipage}{\\wd0}\\scriptsize " + note + "\\end{minipage}\\end{table}}"]
    coverage = {k: {w: float(np.isfinite(vals[k][w]).mean()) for w in MEASURES} for k in vals}
    return rows, coverage, "\n".join(lines), vals


# --- E. sandbox behaviour ------------------------------------------------------------------------

METHODS = {
    "filter": re.compile(r"butter|filtfilt|sosfilt|iirnotch|savgol|medfilt|lfilter|detrend"),
    "notch": re.compile(r"iirnotch|notch"),
    "peaks": re.compile(r"find_peaks|argrelextrema"),
    "median_beat": re.compile(r"median\s*\(|nanmedian"),
    "plot": re.compile(r"matplotlib|plt\."),
}


PRINTED = {  # a measurement label in the printed output of any code run
    "hr": re.compile(r"\b(HR|heart ?rate|rate|bpm)\b", re.I),
    "qrs": re.compile(r"QRS"),
    "pr": re.compile(r"\bPR\b"),
    "qtc": re.compile(r"\bQTc?\b"),
    "axis": re.compile(r"axis", re.I),
    "sokolow": re.compile(r"sokolow|S_?V1\s*\+\s*R_?V|(?<![A-Za-z])(?:Sok|SL)\b\s*[:=,]?\s*-?\d", re.I),  # 'Sok 2.36'
}
# a per-lead amplitude row, also with the value glued to its label ('V1   R0.64 S-0.12 ST0.01 T0.20')
LEAD_AMP = re.compile(r"^\s*(I|II|III|aVR|aVL|aVF|V[1-6])\b.*(?<![A-Za-z])R(?:\b|\s*[=:]?\s*-?\d)"
                      r".*(?<![A-Za-z])S(?:\b|\s*[=:]?\s*-?\d)", re.M)
NUM = re.compile(r"-?\d+\.?\d*")
BUTTER = re.compile(r"butter\(\s*\d+\s*,\s*(\[[^\]]*\]|[\d.]+)((?:[^()]|\([^()]*\))*)\)")
# an uncaught Python exception; a piped command ('python a.py | tail') returns the exit status of its last command.
# BrokenPipeError from '| head' is not an error: the model received the output it asked for.
TRACEBACK = "Traceback (most recent call last)"


def run_error(c):
    if not isinstance(c, dict):
        return False
    stderr = str(c.get("stderr", ""))
    return (c.get("return_code") not in (0, None) or str(c.get("type", "")).endswith("error")
            or (TRACEBACK in stderr and "BrokenPipeError" not in stderr))


def filter_band(code):
    """(low, high, separate): cut-offs in Hz of the first Butterworth band-pass, or of the first scalar high-pass and
    the first scalar low-pass (scipy's default btype) when the band is applied as two filters."""
    lo = hi = None
    for mt in BUTTER.finditer(code):
        wn, rest = mt.group(1), mt.group(2)
        if wn.startswith("["):
            v = re.findall(r"[\d.]+", wn)
            if len(v) == 2:
                return float(v[0]), float(v[1]), False
        elif "high" in rest:
            lo = float(wn) if lo is None else lo
        else:
            hi = float(wn) if hi is None else hi
    return lo, hi, True


def sandbox(run_dir):
    lines = [r for r in results_lines(run_dir) if r.get("type") == "succeeded"]
    c2a, k = custom_to_anon(run_dir), key(run_dir)
    per = []
    for r in lines:
        uses, errors, code_text, tools = 0, 0, [], Counter()
        outs, last_error = [], False
        for b in r.get("content", []):
            if b.get("type") == "server_tool_use":
                uses += 1
                tools[b.get("name")] += 1
                inp = b.get("input", {})
                code_text.append(str(inp.get("command") or inp.get("code") or inp.get("file_text") or ""))
            elif str(b.get("type", "")).endswith("_tool_result"):
                c = b.get("content", {})
                err = run_error(c)
                errors += err
                last_error = err
                if isinstance(c, dict):
                    outs.append(str(c.get("stdout", "")))
        code = "\n".join(code_text)
        out = "\n".join(outs)
        lo, hi, separate = filter_band(code)
        band = lo == 0.5 and hi is not None and hi <= 40
        per.append({"ecg_id": k[c2a[r["custom_id"]]], "uses": uses, "errors": errors, "last_error": last_error,
                    "chars": len(code), "lines": sum(1 for l in code.splitlines() if l.strip()), "tools": dict(tools),
                    "band_05_40": band, "band_separate": band and separate,
                    "dump": any(len(NUM.findall(l)) >= 20 for l in out.splitlines()),
                    "lead_amplitudes": len({mt.group(1) for mt in LEAD_AMP.finditer(out)}) >= 6,
                    **{f"printed_{w}": bool(rx.search(out)) for w, rx in PRINTED.items()},
                    **{k_: bool(rx.search(code)) for k_, rx in METHODS.items()}})
    df = pd.DataFrame(per).set_index("ecg_id")
    df["full_table"] = df[[f"printed_{w}" for w in ("qrs", "pr", "qtc", "axis")]].all(axis=1)
    tools = Counter()
    for d in df["tools"]:
        tools.update(d)
    return {"n": len(df), "uses_mean": float(df["uses"].mean()), "errors_total": int(df["errors"].sum()),
            "records_with_error": float((df["errors"] > 0).mean()),
            "error_rate_per_run": float(df["errors"].sum() / df["uses"].sum()),
            "lines_median": float(df["lines"].median()), "lines_q1": float(df["lines"].quantile(.25)),
            "lines_q3": float(df["lines"].quantile(.75)),
            "records_with_error_n": int((df["errors"] > 0).sum()), "records_last_run_error": int(df["last_error"].sum()),
            "band_05_40": float(df["band_05_40"].mean()), "band_05_40_n": int(df["band_05_40"].sum()),
            "band_separate_n": int(df["band_separate"].sum()),
            "dump": float(df["dump"].mean()), "lead_amplitudes": float(df["lead_amplitudes"].mean()),
            "full_table": float(df["full_table"].mean()),
            **{f"printed_{w}": float(df[f"printed_{w}"].mean()) for w in PRINTED},
            **{f"uses_{k}": float(df[k].mean()) for k in METHODS}, "tools": dict(tools)}, df


# --- F. calibration and F1 -----------------------------------------------------------------------

def calib_bins(p, n_bins=10):
    """Equal-width bins; a probability on an edge (0.3, 0.6, ...) goes to the bin that starts there (rounding absorbs
    floating-point error). The reliability plot in analysis/figures.py uses the same rule."""
    return np.clip(np.floor(np.round(p * n_bins, 9)).astype(int), 0, n_bins - 1)


def ece(y, p, n_bins=10):
    bins = calib_bins(p, n_bins)
    return float(sum((bins == b).sum() / len(p) * abs(y[bins == b].mean() - p[bins == b].mean())
                     for b in range(n_bins) if (bins == b).any()))


def calib_slope(y, p):
    q = np.clip(p, 1e-4, 1 - 1e-4)
    x = np.log(q / (1 - q)).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(x, y)
    return float(lr.intercept_[0]), float(lr.coef_[0, 0])


def calibration(m, pred, ids):
    y = truth(m, ids)
    p = pred.loc[ids, PROB_COLS].to_numpy(float)
    out = {}
    for j, l in enumerate(LABELS):
        a, b = calib_slope(y[:, j], p[:, j])
        out[l] = {"brier": float(np.mean((p[:, j] - y[:, j]) ** 2)), "ece": ece(y[:, j], p[:, j]),
                  "mean_p": float(p[:, j].mean()), "prevalence": float(y[:, j].mean()), "intercept": a, "slope": b}
    return out


# --- G. MI likelihood ----------------------------------------------------------------------------

def mi_likelihood(m, code, agesex, ids):
    """AUROC of p_MI for MI-positive records grouped by the highest likelihood among their MI statements, each group
    against all test records without MI; 95% CIs from stratified bootstrap resamples as for the statements. The
    age-and-sex control's AUROC on the same groups shows how much of a fall any label-trained classifier shows."""
    st = pd.read_csv(m.attrs["root"] + "/scp_statements.csv", index_col=0)
    mi_codes = set(st[(st["diagnostic"] == 1) & (st["diagnostic_class"] == "MI")].index)
    t = m.loc[ids]
    liks = t["scp_codes"].apply(lambda d: [v for c, v in d.items() if c in mi_codes])
    lik = liks.apply(lambda v: max(v, default=np.nan))
    neg = (t["y_MI"] == 0).to_numpy()
    pc, pa = code.loc[ids, "p_MI"].to_numpy(), agesex.loc[ids, "p_MI"].to_numpy()
    rng = np.random.default_rng(RNG_SEED)
    out = {"n_mixed": int(liks.apply(lambda v: len(set(v)) > 1).sum())}   # MI statements with different likelihoods
    for name, sel in [("100", lik == 100), ("50to80", (lik >= 50) & (lik < 100)), ("15to35", (lik > 0) & (lik < 50)),
                      ("zero", lik == 0)]:
        sel = sel.to_numpy()
        if sel.sum() < 10:
            continue
        ip, ineg = np.where(sel)[0], np.where(neg)[0]
        yy = np.r_[np.ones(len(ip)), np.zeros(len(ineg))]
        boots = []
        for _ in range(N_STMT_BOOT):   # positives and negatives resampled separately
            bp, bn = rng.choice(ip, len(ip)), rng.choice(ineg, len(ineg))
            boots.append(auc(yy, np.r_[pc[bp], pc[bn]]))
        out[name] = {"n": int(sel.sum()), "code": auc(yy, np.r_[pc[ip], pc[ineg]]), "ci": ci(boots),
                     "agesex": auc(yy, np.r_[pa[ip], pa[ineg]])}
    return out


# --- main ----------------------------------------------------------------------------------------

def main():
    m = meta()
    code = preds(TEST)
    ids = sorted(code.index)
    mac = {}

    char_rows, char_tex = characteristics(m)
    mac["PHNTrainPatients"] = thousands(char_rows[0]["patients"])
    mac["PHTestAgeMedian"] = f"{char_rows[2]['age_median']:.0f}"
    mac["PHTestFemalePct"] = pct(char_rows[2]["female"], 1)
    mac["PHTestNoisePct"] = pct(char_rows[2]["noise"], 1)
    mac["PHTestAFPct"] = pct(char_rows[2]["afib"], 1)

    sub_rows, sub_tex = subgroups(m, code, ids)
    by = {(r["kind"], r["group"]): r for r in sub_rows}
    for key_, (k, g) in {"Women": ("Sex", "Female"), "Men": ("Sex", "Male"), "AF": ("Rhythm", "Atrial fibrillation or flutter"),
                         "NotAF": ("Rhythm", "Other rhythms"), "Noise": ("Signal quality", "Flag noted"),
                         "NoNoise": ("Signal quality", "No flag"), "Young": ("Age", "Under 50 years"),
                         "AgeMid": ("Age", "50--69 years"), "Old": ("Age", "70 years or older")}.items():
        r = by[(k, g)]
        mac[f"PHSub{key_}N"] = thousands(r["n"])
        if "code" in r:
            mac[f"PHSub{key_}"] = f3(r["code"])
            mac[f"PHSub{key_}Low"] = f3(r["code_ci"][0])
            mac[f"PHSub{key_}High"] = f3(r["code_ci"][1])
    # macro AUROC over MI, STTC, CD and HYP: NORM has few positives in atrial fibrillation or flutter
    for key_, g in [("AF", "Atrial fibrillation or flutter"), ("NotAF", "Other rhythms")]:
        r = by[("Rhythm", g)]
        mac[f"PHSub{key_}NoNorm"] = f3(np.mean([r["per_class"][l] for l in LABELS if l != "NORM"]))
        mac[f"PHSub{key_}NormPos"] = thousands(r["positives"]["NORM"])
    devs = [r for r in sub_rows if r["kind"] == "Device" and "code" in r]
    mac["PHSubNDevices"] = str(len(devs))
    mac["PHSubDeviceRecordsN"] = thousands(sum(r["n"] for r in devs))
    mac["PHSubDeviceMin"] = f3(min(r["code"] for r in devs))
    mac["PHSubDeviceMax"] = f3(max(r["code"] for r in devs))
    mac["PHMinSubgroup"] = str(MIN_SUBGROUP)

    stmt_rows, stmt_tex = statement_table(m, code, ids)
    mac["PHNStatements"] = str(len(stmt_rows))
    mac["PHMinStatement"] = str(MIN_STATEMENT)
    mac["PHNStmtBoot"] = thousands(N_STMT_BOOT)
    for r in stmt_rows:
        nm = re.sub(r"[^A-Za-z]", "", r["statement"].replace("1", "One").replace("_", ""))
        mac[f"PHStmt{nm}"] = f3(r["code"])
        mac[f"PHStmt{nm}Low"] = f3(r["ci"][0])
        mac[f"PHStmt{nm}High"] = f3(r["ci"][1])
        mac[f"PHStmt{nm}N"] = thousands(r["n"])

    meas_rows, coverage, meas_tex, vals = measurement_audit(m, code, ids)
    for k, name in [("code", "Code")]:
        for w in ["hr", "pr", "qrs", "qtc", "axis", "sokolow"]:
            mac[f"PHStated{name}{w.capitalize()}"] = pct(coverage[k][w], 1)  # shares of the whole test fold: one decimal
    tag = {("CLBBB", "qrs"): "CLBBB", ("CRBBB", "qrs"): "CRBBB", ("1AVB", "pr"): "OneAVB", ("SBRAD", "hr"): "SBRAD",
           ("STACH", "hr"): "STACH", ("LAFB", "axis"): "LAFB", ("LVH", "sokolow"): "LVH",
           ("AFIB or AFLT", "af"): "AF", ("SR, no AFIB or AFLT", "af"): "SRAF", ("NORM only", "qrs"): "NormQRS",
           ("Axis field ALAD", "axis"): "ALAD", ("Axis field MID", "axis"): "MID", ("LNGQT", "qtc"): "LNGQT",
           ("IRBBB", "qrs"): "IRBBB", ("NORM only", "sokolow"): "NormSokolow"}
    tag[("IVCD", "qrs")] = "IVCD"
    for r in meas_rows:
        t_ = tag.get((r["reference"], r["measure"]))
        if t_ == "LAFB" and "-45" in r["statistic"]:
            t_ = "LAFBFortyFive"
        if not t_:
            continue
        mac[f"PHMeas{t_}N"] = thousands(r["n"])
        for k, name in [("code", "Code")]:
            few = r["measure"] != "af" and r[f"{k}_n"] < 10   # '--', as in the table
            mac[f"PHMeas{t_}{name}Crit"] = (pct(r[f"{k}_stat"], 1 if r[f"{k}_stat"] < 0.05 else 0)
                                            if np.isfinite(r[f"{k}_stat"]) and not few else "--")
            if r["measure"] != "af":
                mac[f"PHMeas{t_}{name}Stated"] = pct(r[f"{k}_stated"])
                med = r[f"{k}_median"]
                mac[f"PHMeas{t_}{name}Median"] = ("--" if not np.isfinite(med) or few else
                                                   (f"{med:.1f}" if r["measure"] == "sokolow" else f"{med:.0f}"))

    sb, sbdf = sandbox(TEST)
    sbdf = sbdf.loc[ids]
    prov = {}
    for w in ["hr", "qrs", "pr", "qtc", "axis", "sokolow"]:
        st_mask = np.isfinite(vals["code"][w])
        prov[w] = {"stated": float(st_mask.mean()), "printed_any": float(sbdf[f"printed_{w}"].mean()),
                   "printed_among_stated": float(sbdf[f"printed_{w}"].to_numpy()[st_mask].mean())}
        mac[f"PHPrinted{w.capitalize()}Pct"] = pct(prov[w]["printed_any"], 1)
        mac[f"PHPrintedAmongStated{w.capitalize()}Pct"] = pct(prov[w]["printed_among_stated"], 1)
    # records whose stated PR, QTc or axis had no labelled value in the printed output: how many printed lines of
    # samples (the dump detector, a lower bound) from which the value could have been read
    dump = sbdf["dump"].to_numpy()
    for w in ["pr", "qtc", "axis"]:
        unlabelled = np.isfinite(vals["code"][w]) & ~sbdf[f"printed_{w}"].to_numpy()
        prov[w]["unlabelled_n"] = int(unlabelled.sum())
        prov[w]["dump_among_unlabelled"] = float(dump[unlabelled].mean())
        mac[f"PHDumpUnlabelled{w.capitalize()}Pct"] = pct(prov[w]["dump_among_unlabelled"], 1)
    sb["provenance"] = prov
    names_ = {"hr": "Heart rate", "qrs": "QRS duration", "pr": "PR interval", "qtc": "QTc", "axis": "Frontal axis",
              "sokolow": "Sokolow--Lyon voltage"}
    prov_tex = ["\\newcommand{\\ProvenanceTable}{%", "\\begin{table}[!htbp]\\centering",
                "\\caption{\\textbf{Stated measurements and labelled values in the printed code output (test fold, "
                "post hoc).} The ``Stated'' column gives the share of the "
                f"{thousands(len(ids))} records whose key findings state the measurement (regular-expression extraction, "
                "Supplementary Table~\\ref{tab:measurements}). The ``Printed by code'' column gives the share of all "
                "records whose printed code output contains the measurement's label (for example \\texttt{QRS}, "
                "\\texttt{PR}, \\texttt{QT} or \\texttt{QTc}, \\texttt{axis}, \\texttt{Sokolow} or \\texttt{Sok}), and "
                "the last column the same share among the records whose key findings state the measurement. A label can be printed without the value being the one stated, and a value can be "
                "computed and printed under another label (for example a QRS extent printed as onset and offset times), "
                "so the table shows how often a labelled value was printed, not where a stated value came from. Not shown "
                f"in the table: in {pct(sb['dump'], 1)}\\% of records a code run printed at least one line of 20 or more "
                "numbers (typically down-sampled median-beat samples; arrays that numpy wrapped over several shorter "
                f"lines are not counted, so more records printed samples), in {pct(sb['lead_amplitudes'], 1)}\\% a "
                "per-lead table of R and S amplitudes (usually with J-point, J+60~ms and T amplitudes), and in "
                f"{pct(sb['full_table'], 1)}\\% labels for QRS, PR, QT and axis together, as the prompt's measurement "
                "table asked. Of the records whose key findings stated a PR interval, a QTc or an axis with no label "
                f"for it in the printed output, {pct(prov['pr']['dump_among_unlabelled'], 1)}\\%, "
                f"{pct(prov['qtc']['dump_among_unlabelled'], 1)}\\% and {pct(prov['axis']['dump_among_unlabelled'], 1)}\\% "
                "printed such lines of 20 or more numbers.}",
                "\\label{tab:provenance}\\small",
                "\\begin{tabular}{lccc}\\toprule",
                "Measurement & Stated & Printed by code & Printed, among records stating it \\\\ \\midrule"]
    share = lambda x: pct(x, 1)  # noqa: E731   # as the macros: one decimal
    for w in ["hr", "qrs", "pr", "qtc", "axis", "sokolow"]:
        prov_tex.append(f"{names_[w]} & {pct(prov[w]['stated'], 1)}\\% & {share(prov[w]['printed_any'])}\\% & "
                        f"{share(prov[w]['printed_among_stated'])}\\% \\\\")
    prov_tex += ["\\bottomrule\\end{tabular}\\end{table}}"]
    prov_tex = "\n".join(prov_tex)
    mac.update({"PHSandboxRecordsWithErrorN": thousands(sb["records_with_error_n"]),
                "PHSandboxLastRunErrorN": thousands(sb["records_last_run_error"]),
                "PHSandboxRecoveredN": thousands(sb["records_with_error_n"] - sb["records_last_run_error"]),
                "PHSandboxCodeRunsTotal": thousands(round(sb["uses_mean"] * sb["n"])),
                # a 0.5-40 Hz band, as one band-pass or as a 0.5 Hz high-pass and a separate low-pass
                "PHFilterFortyN": thousands(sb["band_05_40_n"]), "PHFilterFortyPct": pct(sb["band_05_40"], 1),
                "PHFilterSeparateN": thousands(sb["band_separate_n"]),
                "PHDumpPct": pct(sb["dump"], 1), "PHLeadAmplitudePct": pct(sb["lead_amplitudes"], 1),
                "PHFullTablePct": pct(sb["full_table"], 1),
                "PHMedianN": thousands(round(sb["uses_median_beat"] * sb["n"])),
                "PHSandboxErrorsTotal": thousands(sb["errors_total"]),
                "PHSandboxRecordsWithErrorPct": pct(sb["records_with_error"], 1),
                "PHSandboxErrorRatePct": pct(sb["error_rate_per_run"], 1),
                "PHSandboxLinesMedian": f"{sb['lines_median']:.0f}",
                "PHSandboxLinesQOne": f"{sb['lines_q1']:.0f}", "PHSandboxLinesQThree": f"{sb['lines_q3']:.0f}",
                "PHSandboxFilterPct": pct(sb["uses_filter"], 1), "PHSandboxNotchPct": pct(sb["uses_notch"]),
                "PHSandboxPeaksPct": pct(sb["uses_peaks"], 1), "PHSandboxMedianPct": pct(sb["uses_median_beat"], 1),
                "PHSandboxPlotPct": pct(sb["uses_plot"], 2)})

    cal_code = calibration(m, code, ids)
    for l in LABELS:
        mac[f"PHECE{l}"] = f3(cal_code[l]["ece"])
        mac[f"PHSlope{l}"] = f"{cal_code[l]['slope']:.2f}"
        mac[f"PHMeanP{l}"] = f"{cal_code[l]['mean_p']:.2f}"
        mac[f"PHPrev{l}"] = f"{cal_code[l]['prevalence']:.2f}"
    mac["PHECEMean"] = f3(np.mean([cal_code[l]["ece"] for l in LABELS]))
    mac["PHBrierMean"] = f3(np.mean([cal_code[l]["brier"] for l in LABELS]))
    mac["PHSlopeMin"] = f"{min(cal_code[l]['slope'] for l in LABELS):.2f}"
    mac["PHSlopeMax"] = f"{max(cal_code[l]['slope'] for l in LABELS):.2f}"
    mac["PHSlopeMinNoCD"] = f"{min(cal_code[l]['slope'] for l in LABELS if l != 'CD'):.2f}"   # CD is near calibrated
    mac["PHSlopeMaxNoCD"] = f"{max(cal_code[l]['slope'] for l in LABELS if l != 'CD'):.2f}"

    mi = mi_likelihood(m, code, preds(AGESEX_TEST), ids)
    mac["PHMILikMixedN"] = thousands(mi["n_mixed"])
    # the same results as the lower part of the subgroup table
    lik_rows = [f"{lab} & {thousands(mi[k]['n'])} & {f3(mi[k]['code'])} [{f3(mi[k]['ci'][0])}, "
                f"{f3(mi[k]['ci'][1])}] & {f3(mi[k]['agesex'])} \\\\"
                for k, lab in [("100", "100"), ("50to80", "50--80"), ("15to35", "15--35"), ("zero", "0")] if k in mi]
    end = "\\bottomrule\\end{tabular}\\end{table}}"
    assert sub_tex.endswith(end)
    sub_tex = sub_tex[:-len(end)] + "\n".join(
        ["\\bottomrule\\end{tabular}", "", "\\medskip",
         "\\begin{tabular}{lrcc}\\toprule",
         "Highest MI likelihood & MI-positive records & MI AUROC [95\\% CI] & Age-and-sex control \\\\ \\midrule",
         *lik_rows, "\\bottomrule\\end{tabular}\\end{table}}"])
    for k, name in [("100", "Hundred"), ("50to80", "Mid"), ("15to35", "Low"), ("zero", "Zero")]:
        if k in mi:
            mac[f"PHMILik{name}N"] = thousands(mi[k]["n"])
            mac[f"PHMILik{name}"] = f3(mi[k]["code"])
            mac[f"PHMILik{name}Low"], mac[f"PHMILik{name}High"] = (f3(x) for x in mi[k]["ci"])
            mac[f"PHMILik{name}AgeSex"] = f3(mi[k]["agesex"])

    # patient-level cluster bootstrap (records of one patient resampled together)
    rng = np.random.default_rng(RNG_SEED)
    pid = m.loc[ids, "patient_id"].to_numpy()
    groups = pd.Series(np.arange(len(ids))).groupby(pid).apply(list).tolist()
    y = truth(m, ids)
    pc_ = code.loc[ids, PROB_COLS].to_numpy(float)
    main_b = []
    while len(main_b) < 2000:
        idx = np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))])
        yy = y[idx]
        if not all(0 < yy[:, j].sum() < len(idx) for j in range(len(LABELS))):
            continue
        main_b.append(np.mean([auc(yy[:, j], pc_[idx, j]) for j in range(len(LABELS))]))
    mac["PHNPatientsTest"] = thousands(len(groups))
    multi = [g for g in groups if len(g) > 1]
    mac["PHNPatientsMulti"] = thousands(len(multi))
    mac["PHNRecordsMulti"] = thousands(sum(len(g) for g in multi))
    mac["PHClusterLow"], mac["PHClusterHigh"] = (f3(x) for x in ci(main_b))

    # how often the summarized thinking refers to PTB-XL by name (the prompt names PTB-XL's superclasses); the API
    # returned an empty summary for many records
    n_think = n_nonempty = n_ptbxl = 0
    for r in results_lines(TEST):
        if r.get("type") == "succeeded":
            n_think += 1
            n_nonempty += bool((r.get("thinking_summary") or "").strip())
            n_ptbxl += bool(re.search(r"PTB-?XL", r.get("thinking_summary") or "", re.I))
    mac["PHSummariesNonEmptyN"] = thousands(n_nonempty)
    mac["PHThinkPTBXLN"] = thousands(n_ptbxl)
    mac["PHThinkPTBXLPct"] = pct(n_ptbxl / n_think, 1)   # of all records
    mac["PHThinkPTBXLNonEmptyPct"] = pct(n_ptbxl / n_nonempty, 1)   # of the non-empty summaries

    write_macros(GEN / "posthoc.tex", mac, "python analysis/posthoc.py (POST HOC and exploratory analyses)")
    with open(GEN / "posthoc.tex", "a") as f:
        f.write("\n".join([char_tex, sub_tex, stmt_tex, meas_tex, prov_tex]) + "\n")
    write_json({"characteristics": char_rows, "subgroups": sub_rows, "statements": stmt_rows,
                "measurements": meas_rows, "coverage": coverage, "sandbox": sb, "calibration": cal_code,
                "mi_likelihood": mi}, OUT_JSON / "posthoc_results.json")
    print("coverage", {k: {w: round(v, 2) for w, v in c.items()} for k, c in coverage.items()})
    print("ECE", mac["PHECEMean"], "slopes", mac["PHSlopeMin"], mac["PHSlopeMax"], "MI lik", mi)
    for r in sub_rows:
        print("sub", r["kind"], r["group"], r["n"], round(r.get("code", np.nan), 3))
    for r in stmt_rows:
        print("stmt", r["class"], r["statement"], r["n"], round(r["code"], 3), [round(x, 3) for x in r["ci"]])


if __name__ == "__main__":
    main()
