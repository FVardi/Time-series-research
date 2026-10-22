"""Tables, figures and cited numbers for the results document (report/results.tex).

Usage (from the repository root):
    uv run python scripts/make_report.py

Reads the run folders named below (runs/ is not committed) and writes, into report/
(committed):
    tables/*.tex        LaTeX tables, included with \\input
    tables/numbers.tex  \\newcommand definitions for every number quoted in the prose
    figures/*.pdf       figures, included with \\includegraphics
Every number in the document comes from here; none is typed by hand (D98).

The document reports current results only, organised by topic (D98). Every run folder in
runs/ is listed in the run index (RUN_INDEX, appendix of the document), including runs
that are superseded or not written up; a folder missing from RUN_INDEX is listed as
"not indexed" and a warning is printed.

Needs matplotlib in addition to the project's dependencies.
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")  # files only, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Run folders behind each part of the document. The keys are only used in this script.
# ---------------------------------------------------------------------------
RUNS = {
    # TF-C on FD-A -> FD-B
    "tfc_full_1": "runs/20261008_134951_tfc_official_fd",  # official, full data: first attempt, seed 0 only
    "tfc_full_2": "runs/20261008_143021_tfc_official_fd",  # second attempt, all seeds
    "tfc_full_3": "runs/20261008_155516_tfc_official_fd",  # third attempt, after the D44 change
    "tfc_ours": "runs/20261008_142751_tfc_fd",
    "tfc_probe": "runs/20261008_151014_probe_tfc_fd",
    "fdb_spectrum": "runs/20261008_162447_probe_spectrum_fd",
    "tfc_subset": "runs/20261008_171159_tfc_official_fd_subset",
    # Herning
    "band_check": "runs/20261009_085307_band_check_herning",
}

# FordA: SSL methods and baselines (D75-D80, D85, D89, D90). TS2Vec, T-Loss and the baselines were run
# on 2026-10-20 with timing (D79); TS-JEPA (2026-10-16) and PatchTST (2026-10-19) earlier, without timing
# (D102, D103). TS-JEPA in its masking-fixed version (D102); PatchTST with mean pooling (D87).
FORDA_SSL = {
    "TS2Vec": "runs/20261020_144711_ts2vec_forda",
    "T-Loss": "runs/20261020_144932_tloss_forda",
    "TS-JEPA (masking fixed)": "runs/20261016_114616_tsjepa_forda_fixed",
    "PatchTST": "runs/20261019_154248_patchtst_forda_mean",
}
FORDA_NOT_TIMED = ["TS-JEPA (masking fixed)", "PatchTST"]  # shown as "not timed" in the timing table (D103)
# Short names of the SSL methods in the \\Forda... numbers quoted in the prose.
FORDA_KEYS = {"TS2Vec": "Tsvec", "T-Loss": "Tloss", "TS-JEPA (masking fixed)": "Tsjepa", "PatchTST": "Patchtst"}
# TS-JEPA's reproduction check uses the official-masking run, the paper's code (D103); not in the comparison (D102).
FORDA_TSJEPA_OFFICIAL = "runs/20261016_114600_tsjepa_forda"
FORDA_BASELINES = {
    "Raw samples": "runs/20261020_143928_raw_forda",  # also holds the floor probe (D83)
    "|FFT|": "runs/20261020_144122_fft_forda",
    "catch24": "runs/20261020_144235_catch22_forda",
    "MiniRocket": "runs/20261020_152809_minirocket_forda",
}
# The runs these replaced (quoted in one sentence: run-to-run variation).
FORDA_EARLIER = {
    "TS2Vec": "runs/20261008_084722_ts2vec_forda",
    "T-Loss": "runs/20261008_092321_tloss_forda",
}

# ---------------------------------------------------------------------------
# Run index (D98): every folder in runs/ with what it was for and its status.
# Status: "In <section label>" for a reported run; "Superseded by <folder>"; "Empty";
# "Not written up yet"; "Examination" (one-off dataset checks, cited where used).
# ---------------------------------------------------------------------------
RUN_INDEX = {
    "20261007_214220_ts2vec_forda_cpu_check": ("TS2Vec on FordA, CPU check in the cloud workspace",
                                               "Superseded by 20261008_084722_ts2vec_forda"),
    "20261008_084722_ts2vec_forda": ("TS2Vec on FordA (reproduction)",
                                     "Superseded by 20261020_144711_ts2vec_forda (rerun with timing)"),
    "20261008_092321_tloss_forda": ("T-Loss on FordA (reproduction)",
                                    "Superseded by 20261020_144932_tloss_forda (rerun with timing)"),
    "20261008_134951_tfc_official_fd": ("Official TF-C, full FD-A data, attempt 1", "In \\S\\ref{tfc-full}"),
    "20261008_142751_tfc_fd": ("Our TF-C, FD-A to FD-B", "In \\S\\ref{tfc-ours}"),
    "20261008_143021_tfc_official_fd": ("Official TF-C, full FD-A data, attempt 2", "In \\S\\ref{tfc-full}"),
    "20261008_151014_probe_tfc_fd": ("Frozen probes on our TF-C encoders", "In \\S\\ref{tfc-probe}"),
    "20261008_155516_tfc_official_fd": ("Official TF-C, full FD-A data, attempt 3", "In \\S\\ref{tfc-full}"),
    "20261008_162447_probe_spectrum_fd": ("Probes on the |FFT| spectra of FD-B", "In \\S\\ref{fdb-spectrum}"),
    "20261008_171159_tfc_official_fd_subset": ("Official TF-C, debugging subset", "In \\S\\ref{tfc-subset}"),
    "20261009_085150_band_check_herning": ("Herning band check, stopped before writing results", "Empty"),
    "20261009_085307_band_check_herning": ("Herning band check", "In \\S\\ref{band-check}"),
    "20261012_205347_spectra_herning": ("Herning spectra", "Examination (QUESTIONS.md, Q3)"),
    "20261012_210217_vfd_lines_herning": ("Frequency-converter lines in the Herning spectra",
                                          "Examination (QUESTIONS.md, Q3)"),
    "20261014_herning_examination": ("Herning dataset examination", "Examination (eda/herning)"),
    "20261016_114600_tsjepa_forda": ("TS-JEPA on FordA (official masking)",
                                     "In \\S\\ref{forda-repro} only; not in the comparison (D102)"),
    "20261016_114616_tsjepa_forda_fixed": ("TS-JEPA (masking fixed) on FordA", "In \\S\\ref{forda-baselines}"),
    "20261017_115419_collapse_forda": ("Collapse check on FordA", "Not written up yet"),
    "20261019_130910_patchtst_forda": ("PatchTST on FordA (last-patch pooling)",
                                       "Superseded by 20261019_154248_patchtst_forda_mean (same models, D87)"),
    "20261019_154248_patchtst_forda_mean": ("PatchTST re-encoded with mean pooling (D86, D87)",
                                            "In \\S\\ref{forda-baselines}"),
    "20261019_154305_patchtst_forda_flatten": ("PatchTST re-encoded with flatten pooling (D86)", "Not written up yet"),
    "20261020_135846_ablation_forda_A": ("TF-C ablation on FordA, step A", "Not written up yet"),
    "20261020_140639_ablation_forda_B": ("TF-C ablation on FordA, step B", "Not written up yet"),
    "20261020_141430_ablation_forda_C": ("TF-C ablation on FordA, step C", "Not written up yet"),
    "20261020_143928_raw_forda": ("Baseline: raw samples (and floor) on FordA", "In \\S\\ref{forda-baselines}"),
    "20261020_144122_fft_forda": ("Baseline: |FFT| on FordA", "In \\S\\ref{forda-baselines}"),
    "20261020_144235_catch22_forda": ("Baseline: catch24 on FordA", "In \\S\\ref{forda-baselines}"),
    "20261020_144711_ts2vec_forda": ("TS2Vec on FordA, with timing", "In \\S\\ref{forda-repro}, \\S\\ref{forda-baselines}"),
    "20261020_144932_tloss_forda": ("T-Loss on FordA, with timing", "In \\S\\ref{forda-repro}, \\S\\ref{forda-baselines}"),
    "20261020_152809_minirocket_forda": ("Baseline: MiniRocket on FordA", "In \\S\\ref{forda-repro}, \\S\\ref{forda-baselines}"),
}

# Published numbers that are not in the run configs: standard deviations of the TF-C paper
# (arXiv 2206.08496, Table 5, FD-A -> FD-B, 5 runs), as recorded in D38.
TFC_PUBLISHED = {"accuracy": 0.8934, "accuracy_sd": 0.0379, "macro_f1": 0.9162, "macro_f1_sd": 0.0826}

# Herning feature bands and downsampled Nyquist frequencies, as decided (for shading in the figure).
AE_BAND_HZ = (50e3, 560e3)  # D61
US_BAND_HZ = (1e3 * 10 ** -0.15, 126e3 * 10 ** 0.15)  # D60: base-10 octaves, centres 1.0 ... 126 kHz
AE_NYQUIST_HZ = 1.25e6 / 2  # D62: AE downsampled by 2
US_NYQUIST_HZ = 0.5e6 / 2  # D62: US downsampled by 5

# Colours: light-mode categorical slots 1-6 of the validated reference palette (dataviz), and ink.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"
VIOLET, BROWN = "#7a5bd1", "#9c6b30"
INK, MUTED, GRID = "#0b0b0b", "#8a8984", "#e4e3df"

ROOT = "."
OUT = "report"
NUMBERS = {}  # name -> text, written to tables/numbers.tex


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def path(*parts):
    return os.path.join(ROOT, *parts)


def read_csv(run, name, **kw):
    return pd.read_csv(path(run, name), **kw)


def read_yaml(run):
    with open(path(run, "config.yaml")) as f:
        return yaml.safe_load(f)


def read_json(run, name):
    with open(path(run, name)) as f:
        return json.load(f)


def f3(x):
    """Three decimals, or an en dash for a missing value."""
    return "--" if pd.isna(x) else f"{x:.3f}"


def pm(mean, sd):
    """'mean ± sd' in math mode, three decimals."""
    return f"${mean:.3f} \\pm {sd:.3f}$"


def signed(x):
    """Signed difference with three decimals, real minus sign (math mode)."""
    return f"${x:+.3f}$"


def tex(text):
    """Escape the characters of plain text that LaTeX treats specially."""
    return text.replace("\\", "\\textbackslash{}").replace("_", "\\_").replace("|", "$|$").replace("%", "\\%")


def number(name, text):
    """Record a number quoted in the prose as \\name (letters only, LaTeX rule)."""
    assert name.isalpha(), name
    NUMBERS[name] = text


def reproduction(acc, published):
    """Reproduction criterion as implemented in scripts/evaluate.py and scripts/tfc_official.py
    (D26 item 6): |published - mean| <= max(2 SD, 0.01). SD over seeds with ddof = 1 (pandas)."""
    mean, sd = acc.mean(), acc.std()
    allowed = max(2 * sd, 0.01)
    return {"mean": mean, "sd": sd, "n": len(acc), "diff": abs(published - mean), "allowed": allowed,
            "pass": abs(published - mean) <= allowed}


def write_table(name, header, rows, colspec, caption, label, notes=None):
    """A booktabs table: header is a list of column titles, rows a list of lists of strings.
    A row that is the string 'MIDRULE' becomes \\midrule."""
    lines = ["\\begin{table}[htbp]", "\\centering", "\\small", f"\\caption{{{caption}}}",
             f"\\label{{{label}}}", "\\begin{adjustbox}{max width=\\linewidth}",
             f"\\begin{{tabular}}{{{colspec}}}", "\\toprule"]
    lines += [" & ".join(h) + " \\\\" for h in (header if isinstance(header[0], list) else [header])]
    lines.append("\\midrule")
    for r in rows:
        lines.append("\\midrule" if r == "MIDRULE" else " & ".join(r) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{adjustbox}"]
    if notes:
        lines.append(f"\\par\\vspace{{2pt}}\\parbox{{0.95\\linewidth}}{{\\footnotesize {notes}}}")
    lines.append("\\end{table}")
    with open(os.path.join(OUT, "tables", f"{name}.tex"), "w", encoding="utf-8") as f:
        f.write("% Generated by scripts/make_report.py - do not edit by hand.\n" + "\n".join(lines) + "\n")


def style(ax):
    """Recessive axes and grid."""
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=7)


def save(fig, name):
    fig.savefig(os.path.join(OUT, "figures", f"{name}.pdf"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# FordA: reproductions, and SSL methods against the baselines (D75-D80)
# ---------------------------------------------------------------------------
PROBE_NAMES = {"logreg": "Logistic regression", "svm_ts2vec": "SVM (TS2Vec protocol)"}
FRACTIONS = [0.01, 0.1, 1.0]


def pct(frac):
    return f"{frac:.0%}".replace("%", "\\,\\%")


def forda_results():
    """{method: results.csv} for the SSL methods and baselines."""
    return {m: read_csv(run, "results.csv") for m, run in {**FORDA_SSL, **FORDA_BASELINES}.items()}


def forda_accuracy_table(res):
    """Mean ± SD accuracy per method, probe and label budget; best mean per column in bold."""
    floor = res["Raw samples"][res["Raw samples"].probe == "floor"]
    cols = [(p, f) for p in PROBE_NAMES for f in FRACTIONS]
    cell = {}  # (method, probe, frac) -> (mean, sd)
    for m, r in res.items():
        for p, f in cols:
            g = r[(r.probe == p) & (r.label_fraction == f)].accuracy
            if len(g):
                cell[(m, p, f)] = (g.mean(), g.std())
    best = {c: max(v[0] for k, v in cell.items() if (k[1], k[2]) == c) for c in cols}

    def fmt(m, p, f):
        if (m, p, f) not in cell:
            return "not run"
        mean, sd = cell[(m, p, f)]
        text = f"{mean:.3f} \\pm {sd:.3f}"
        return f"$\\mathbf{{{mean:.3f}}} \\pm {sd:.3f}$" if np.isclose(mean, best[(p, f)]) else f"${text}$"

    header = [["", "\\multicolumn{3}{c}{Logistic regression}", "\\multicolumn{3}{c}{SVM (TS2Vec protocol)}"],
              ["Method"] + [pct(f) for f in FRACTIONS] * 2]
    rows = []
    for f_group, methods in (("ssl", FORDA_SSL), ("base", FORDA_BASELINES)):
        for m in methods:
            rows.append([tex(m)] + [fmt(m, p, f) for p, f in cols])
        rows.append("MIDRULE")
    fl = [f"${floor[floor.label_fraction == f].accuracy.mean():.3f}$" for f in FRACTIONS]
    rows.append(["Floor (most frequent class)"] + fl + fl)
    write_table("forda_accuracy", header, rows, "lcccccc",
                "FordA test accuracy, frozen representations: mean $\\pm$ SD over 3 seeds; the same "
                "labelled series for every method at each budget (D78). Bold: best mean per column. "
                "Raw samples are not probed with logistic regression (D89). The floor ignores the "
                "features, so it is the same under both probes (D83).", "tab:forda-accuracy")
    return cell


def forda_paired_table(res):
    """Per probe and budget: the best baseline (by mean test accuracy, D80) and each SSL method's
    paired difference to it, per seed (same seeds, same label subsets, D78)."""
    rows, out = [], {}
    for p, pname in PROBE_NAMES.items():
        for f in FRACTIONS:
            means = {m: res[m][(res[m].probe == p) & (res[m].label_fraction == f)].accuracy.mean()
                     for m in FORDA_BASELINES}
            means = {m: v for m, v in means.items() if not np.isnan(v)}
            b = max(means, key=means.get)
            base = res[b][(res[b].probe == p) & (res[b].label_fraction == f)].set_index("seed").accuracy
            row = [pname if f == FRACTIONS[0] else "", pct(f), f"{tex(b)} (${means[b]:.3f}$)"]
            for m in FORDA_SSL:
                ssl = res[m][(res[m].probe == p) & (res[m].label_fraction == f)].set_index("seed").accuracy
                d = (ssl - base).dropna()
                out[(m, p, f)] = d
                row.append(f"{signed(d.mean())} \\,$({', '.join(f'{x:+.3f}' for x in d)})$")  # math mode: real minus signs
            rows.append(row)
        rows.append("MIDRULE")
    write_table("forda_paired", ["Probe", "Labels", "Best baseline (mean)"] +
                [f"{m}: mean (per seed)" for m in FORDA_SSL], rows[:-1], "lll" + "c" * len(FORDA_SSL),
                "FordA: paired difference in test accuracy, SSL method minus the best baseline at that "
                "probe and label budget (positive: the SSL method is better). Per seed, the two use the "
                "same labelled series (D78). The best baseline is chosen by its mean test accuracy, which "
                "favours the baselines (D80). Seeds in the order 0, 1, 2.", "tab:forda-paired")
    return out


def forda_timing_table():
    """Wall-clock time per stage (D79): median over seeds, with the range."""
    rows = []
    for m, run in {**FORDA_SSL, **FORDA_BASELINES}.items():
        if m in FORDA_NOT_TIMED:
            rows.append([tex(m), "--", "not timed", "--", "--", "--"])
            continue
        seeds = sorted(d for d in os.listdir(path(run)) if d.startswith("seed_"))
        t = pd.DataFrame([read_json(run, f"{s}/timing.json") for s in seeds])
        hw = read_json(run, "hardware.json")
        device = "GPU" if hw["device"].startswith("cuda") else "CPU"
        res = read_csv(run, "results.csv")
        probe_s = []
        for p in PROBE_NAMES:
            g = res[(res.probe == p) & (res.label_fraction == 1.0)].probe_fit_s
            probe_s.append(f"{g.median():.1f}" if len(g) else "--")
        fit = t.pretrain_s
        fit_text = "none" if fit.max() < 1e-3 else f"{fit.median():.2f} ({fit.min():.2f}--{fit.max():.2f})"
        rows.append([tex(m), device, fit_text, f"{t.encode_series_per_s.median():,.0f}"] + probe_s)
    compile_s = read_json(FORDA_BASELINES["MiniRocket"], "compile.json")["compile_s"]
    hw = read_json(FORDA_SSL["TS2Vec"], "hardware.json")
    write_table("forda_timing", [["", "", "Fit / pretraining (s)", "Encoding", "\\multicolumn{2}{c}{Probe fit, 100\\,\\% labels (s)}"],
                                 ["Method", "Device", "median (range)", "series/s", "Logistic", "SVM"]],
                rows, "llrrrr",
                f"FordA: wall-clock time per stage (D79), median over 3 seeds. Encoding: training and test "
                f"series (4,921) per second. Probe fit includes the probe's cross-validation. Hardware: "
                f"{hw['gpu']} GPU, {hw['cpu_count']}-thread CPU. MiniRocket's code compiles once on import "
                f"({compile_s:.2f}~s, not included, D81). TS-JEPA and PatchTST were run before timing existed "
                f"(D102, D103).", "tab:forda-timing")


def forda_figure(res):
    """Accuracy against label budget, one panel per probe; SSL methods solid, baselines dashed."""
    colours = dict(zip(list(FORDA_SSL) + list(FORDA_BASELINES), (BLUE, ORANGE, VIOLET, BROWN, GREEN, AQUA, YELLOW, MAGENTA)))
    floor = res["Raw samples"][res["Raw samples"].probe == "floor"].accuracy.mean()
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.7), sharey=True)
    for ax, (p, pname) in zip(axes, PROBE_NAMES.items()):
        for m, r in res.items():
            g = r[r.probe == p].groupby("label_fraction").accuracy
            if not len(g.mean()):
                continue  # raw samples: no logistic probe (D89)
            ssl = m in FORDA_SSL
            ax.errorbar(np.log10(g.mean().index * 100), g.mean(), yerr=g.std(), color=colours[m],
                        linewidth=2, linestyle="-" if ssl else "--", marker="o" if ssl else "s",
                        markersize=4, capsize=2, label=m)
        ax.axhline(floor, color=MUTED, linestyle=":", linewidth=1)
        ax.text(2.18, floor + 0.008, "floor", fontsize=6, color=INK, ha="right")
        ax.set_xticks([0, 1, 2], ["1 %", "10 %", "100 %"])
        ax.set_xlim(-0.15, 2.2)
        ax.set_title(pname, fontsize=8, color=INK, loc="left")
        ax.set_xlabel("Training labels used by the probe", fontsize=7)
        style(ax)
    axes[0].set_ylabel("Test accuracy", fontsize=7)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7, frameon=False, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    save(fig, "forda_label_efficiency")


def forda():
    """All FordA tables, the figure and the numbers quoted in the prose. Returns the reproduction checks."""
    res = forda_results()
    cell = forda_accuracy_table(res)
    paired = forda_paired_table(res)
    forda_timing_table()
    forda_figure(res)

    # Numbers quoted in the prose.
    for m, key in (("TS2Vec", "Tsvec"), ("T-Loss", "Tloss")):
        earlier = read_csv(FORDA_EARLIER[m], "results.csv")
        e = earlier[(earlier.probe == "svm_ts2vec") & (earlier.label_fraction == 1.0)].accuracy
        number(f"Forda{key}Earlier", f3(e.mean()))
    for m, key in FORDA_KEYS.items():
        for p, pk in (("logreg", "Log"), ("svm_ts2vec", "Svm")):
            for f, fk in zip(FRACTIONS, ("One", "Ten", "Hundred")):
                d = paired[(m, p, f)]
                number(f"Forda{key}Diff{pk}{fk}", signed(d.mean()))
                number(f"Forda{key}Neg{pk}{fk}", str(int((d < 0).sum())))
    for m, key in (("TS2Vec", "Tsvec"), ("T-Loss", "Tloss"), ("MiniRocket", "Minirocket")):
        run = {**FORDA_SSL, **FORDA_BASELINES}[m]
        seeds = sorted(d for d in os.listdir(path(run)) if d.startswith("seed_"))
        t = pd.DataFrame([read_json(run, f"{s}/timing.json") for s in seeds])
        number(f"Forda{key}Fit", f"{t.pretrain_s.median():.2f}~s" if t.pretrain_s.median() < 60
               else f"{t.pretrain_s.median() / 60:.1f}~min")
    # First GPU seed vs the others (start-up time is included in the first, D79).
    run = FORDA_SSL["TS2Vec"]
    t = [read_json(run, f"seed_{s}/timing.json")["pretrain_s"] for s in range(3)]
    number("FordaTsvecFitFirst", f"{t[0]:.0f}~s")
    number("FordaTsvecFitRest", f"{min(t[1:]):.0f}--{max(t[1:]):.0f}~s")
    mr = res["MiniRocket"]
    ridge = mr[(mr.probe == "ridge_minirocket") & (mr.label_fraction == 1.0)].accuracy
    number("FordaMinirocketRidge", pm(ridge.mean(), ridge.std()))
    number("FordaFftLogTen", f"{cell[('|FFT|', 'logreg', 0.1)][0]:.3f}")
    number("FordaTsvecLogTen", f"{cell[('TS2Vec', 'logreg', 0.1)][0]:.3f}")

    # Reproduction checks: each method's own probe with all labels against the published accuracy.
    checks = {}
    for m, run in (("TS2Vec", FORDA_SSL["TS2Vec"]), ("T-Loss", FORDA_SSL["T-Loss"]),
                   ("MiniRocket", FORDA_BASELINES["MiniRocket"]),
                   ("TS-JEPA, official masking", FORDA_TSJEPA_OFFICIAL)):
        ev = read_yaml(run)["evaluation"]
        r = read_csv(run, "results.csv")
        acc = r[(r.probe == ev["reproduction_probe"]) & (r.label_fraction == 1.0)].accuracy
        checks[m] = (m, "FordA", ev["published_accuracy"], acc, "\\ref{forda-repro}")
    return checks


# ---------------------------------------------------------------------------
# TF-C on FD-A -> FD-B
# ---------------------------------------------------------------------------
def pretrain_losses(run):
    """{seed: loss per epoch (NaN where not finite / not written)} for a TF-C run."""
    seeds = sorted(d for d in os.listdir(path(run)) if d.startswith("seed_"))
    return {int(s.split("_")[1]): read_csv(run, f"{s}/pretrain_loss.csv") for s in seeds}


def tfc_official_full():
    """Official TF-C on the full data. Per seed: how far pretraining got, and whether
    fine-tuning wrote anything."""
    losses = pretrain_losses(RUNS["tfc_full_3"])
    # The second and third attempts ran the same settings; check the loss curves are identical.
    other = pretrain_losses(RUNS["tfc_full_2"])
    identical = all(losses[s].loss.equals(other[s].loss) for s in losses)
    first = pretrain_losses(RUNS["tfc_full_1"])[0]
    # The first attempt (older code) kept writing after the loss became non-finite: compare finite epochs.
    identical_first = first.loss.dropna().reset_index(drop=True).equals(losses[0].loss.dropna().reset_index(drop=True))
    number("TfcFullIdentical", "identical" if identical else "not identical")
    number("TfcFullFirstIdentical", "identical" if identical_first else "not identical")

    rows = []
    for seed, df in losses.items():
        finite = df.loss.notna()
        n_ok = int(finite.sum())
        first_bad = "--" if n_ok == len(df) else str(int(df.epoch[~finite].iloc[0]))
        finetuned = os.path.exists(path(RUNS["tfc_full_3"], f"seed_{seed}", "epochs.csv"))
        rows.append([str(seed), str(len(df)), str(n_ok), first_bad, f3(df.loss[finite].iloc[-1]),
                     "yes" if finetuned else "no"])
    n_div = sum(r[3] != "--" for r in rows)
    number("TfcFullDiverged", str(n_div))
    number("TfcFullSeeds", str(len(rows)))
    number("TfcFullResultsWritten", "yes" if os.path.exists(path(RUNS["tfc_full_3"], "results.csv")) else "no")
    write_table("tfc_official_full", ["Seed", "Epochs run", "Finite", "First non-finite", "Last finite loss",
                                      "Fine-tuning output"], rows, "rccccc",
                "Official TF-C, full FD-A pretraining data (third attempt). Pretraining loss "
                "per epoch is empty in \\texttt{pretrain\\_loss.csv} once it is no longer finite.",
                "tab:tfc-official-full",
                notes=f"Second and third attempt: pretraining losses {'identical' if identical else 'NOT identical'} "
                      "for every seed.")
    return losses


def tfc_official_subset():
    """Official TF-C with the debugging subset."""
    run = RUNS["tfc_subset"]
    res = read_csv(run, "results.csv")
    rows = []
    for seed, g in res.groupby("seed"):
        ep = read_csv(run, f"seed_{seed}/epochs.csv")
        # Macro recall exactly 1/3 at every epoch = what a constant (single-class) prediction gives.
        const = np.allclose(ep.test_recall, 1 / 3) and np.allclose(ep.val_recall, 1 / 3)
        o, c = g[g.protocol == "official"].iloc[0], g[g.protocol == "clean"].iloc[0]
        rows.append([str(seed), str(int(o.epoch)), f3(o.accuracy), f3(o.macro_f1), str(int(c.epoch)),
                     f3(c.accuracy), f3(c.macro_f1), "yes" if const else "no"])
    write_table("tfc_official_subset",
                ["Seed", "\\multicolumn{3}{c}{Official protocol}", "\\multicolumn{3}{c}{Clean protocol}",
                 "Recall $=1/3$"],
                [["", "Epoch", "Acc.", "F1", "Epoch", "Acc.", "F1", "all epochs"], "MIDRULE"] + rows,
                "rccccccc",
                "Official TF-C with the debugging subset. Official protocol: epoch with the best "
                "test accuracy; clean protocol: epoch with the best validation macro-F1 (D43). Last "
                "column: macro recall on test and validation is exactly $1/3$ at every one of the "
                "40 fine-tuning epochs.", "tab:tfc-official-subset")
    acc = res[res.protocol == "official"].accuracy
    number("TfcSubsetAcc", pm(acc.mean(), acc.std()))
    f1 = res[res.protocol == "official"].macro_f1
    number("TfcSubsetFone", pm(f1.mean(), f1.std()))
    return res


def tfc_ours():
    """Our TF-C."""
    run = RUNS["tfc_ours"]
    res = read_csv(run, "results.csv")
    cols = ["accuracy", "precision", "recall", "f1", "auroc", "auprc"]
    rows = [[str(r.seed), str(int(r.epoch))] + [f3(getattr(r, c)) for c in cols] for r in res.itertuples()]
    rows += ["MIDRULE", ["Mean $\\pm$ SD", ""] + [pm(res[c].mean(), res[c].std()) for c in cols]]
    write_table("tfc_ours", ["Seed", "Epoch", "Acc.", "Prec.", "Recall", "F1", "AUROC", "AUPRC"], rows,
                "rrcccccc", "Our TF-C (D41, D42), FD-A $\\rightarrow$ FD-B test set. Epoch: "
                "fine-tuning epoch with the best validation macro-F1; the test set is scored once, "
                "for that epoch. Precision, recall and F1 are macro averages.", "tab:tfc-ours")
    number("TfcOursAcc", pm(res.accuracy.mean(), res.accuracy.std()))
    number("TfcOursFone", pm(res.f1.mean(), res.f1.std()))
    number("TfcOursAccMin", f3(res.accuracy.min()))
    number("TfcOursAccMax", f3(res.accuracy.max()))

    # Figure: validation macro-F1 per fine-tuning epoch, selected epoch marked.
    fig, ax = plt.subplots(figsize=(6.3, 2.2))
    for r in res.itertuples():
        ep = read_csv(run, f"seed_{r.seed}/epochs.csv")
        ax.plot(ep.epoch, ep.val_f1, color=BLUE, linewidth=1, alpha=0.6)
        ax.scatter([r.epoch], ep.val_f1[ep.epoch == r.epoch], color=BLUE, s=18, zorder=3,
                   edgecolors="white", linewidths=1)
    ax.set_xlabel("Fine-tuning epoch", fontsize=7)
    ax.set_ylabel("Validation macro-F1", fontsize=7)
    ax.set_title("Our TF-C: validation macro-F1 per epoch, one line per seed; dot = selected epoch",
                 fontsize=8, color=INK, loc="left")
    style(ax)
    save(fig, "tfc_ours_validation")
    n_val = int((read_csv(run, "windows.csv").split == "val").sum())
    number("TfcOursNval", str(n_val))
    return res


def tfc_losses_figure(full):
    """Pretraining loss per epoch: official full data, official subset, ours."""
    panels = [("Official TF-C, full data", full),
              ("Official TF-C, debugging subset", pretrain_losses(RUNS["tfc_subset"])),
              ("Our TF-C", pretrain_losses(RUNS["tfc_ours"]))]
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.2))
    for ax, (title, losses) in zip(axes, panels):
        for seed, df in losses.items():
            ax.plot(df.epoch, df.loss, color=BLUE, linewidth=1, alpha=0.7)
            if df.loss.isna().any():  # mark where the loss stopped being finite
                last = df[df.loss.notna()].iloc[-1]
                ax.scatter([last.epoch], [last.loss], marker="x", color=ORANGE, s=20, zorder=3)
        ax.set_title(title, fontsize=7, color=INK, loc="left")
        ax.set_xlabel("Pretraining epoch", fontsize=7)
        style(ax)
    axes[0].set_ylabel("Pretraining loss", fontsize=7)
    axes[0].text(0.98, 0.95, "× = last finite epoch", transform=axes[0].transAxes, ha="right", va="top",
                 fontsize=6, color=INK)
    fig.tight_layout()
    save(fig, "tfc_pretrain_loss")


def fdb_overview(subset_res, ours_res):
    """The TF-C results and the FD-B diagnostics side by side on the FD-B test set."""
    probes = read_csv(RUNS["tfc_probe"], "results.csv")
    spectrum = read_csv(RUNS["fdb_spectrum"], "results.csv")
    name = {"svm_ts2vec": "SVM", "logreg": "log. reg."}
    entries = []  # (label, accuracies per seed, macro-F1 per seed, entry)
    official = subset_res[subset_res.protocol == "official"]
    entries.append(("Official TF-C, subset, fine-tuned", official.accuracy, official.macro_f1, "\\ref{tfc-subset}"))
    entries.append(("Our TF-C, fine-tuned", ours_res.accuracy, ours_res.f1, "\\ref{tfc-ours}"))
    for enc, enc_name in (("pretrained", "pretrained TF-C encoder"), ("random_init", "untrained encoder")):
        for probe in ("svm_ts2vec", "logreg"):
            g = probes[(probes.encoder == enc) & (probes.probe == probe)]
            entries.append((f"{name[probe]} on {enc_name}", g.accuracy, g.macro_f1, "\\ref{tfc-probe}"))
    for probe in ("svm_ts2vec", "logreg"):
        g = spectrum[spectrum.probe == probe]
        entries.append((f"{name[probe]} on $|$FFT$|$ spectrum", g.accuracy, g.macro_f1, "\\ref{fdb-spectrum}"))

    rows = [["TF-C paper (published)", "--", pm(TFC_PUBLISHED["accuracy"], TFC_PUBLISHED["accuracy_sd"]),
             pm(TFC_PUBLISHED["macro_f1"], TFC_PUBLISHED["macro_f1_sd"]), "5"], "MIDRULE"]
    for label, acc, f1, entry in entries:
        cell = (lambda s: f3(s.iloc[0])) if len(acc) == 1 else (lambda s: pm(s.mean(), s.std()))
        rows.append([label, entry, cell(acc), cell(f1), str(len(acc))])
    write_table("fdb_overview", ["Model / probe", "Section", "Accuracy", "Macro-F1", "Seeds"], rows, "llccr",
                "FD-B test set: TF-C results and the diagnostics. Probes use all 60 FD-B "
                "training labels (D45, D46). The $|$FFT$|$ probes are deterministic (one value).",
                "tab:fdb")
    for enc, key in (("pretrained", "Pre"), ("random_init", "Rand")):
        for probe, pkey in (("svm_ts2vec", "Svm"), ("logreg", "Logreg")):
            g = probes[(probes.encoder == enc) & (probes.probe == probe)].accuracy
            number(f"TfcProbe{key}{pkey}", pm(g.mean(), g.std()))
    for probe, pkey in (("svm_ts2vec", "Svm"), ("logreg", "Logreg")):
        number(f"FdbSpec{pkey}", f3(spectrum[spectrum.probe == probe].accuracy.iloc[0]))

    # Figure: one row per model/probe, a dot per seed, a bar at the mean, published band shaded.
    fig, ax = plt.subplots(figsize=(6.3, 2.8))
    pub, sd = TFC_PUBLISHED["accuracy"], TFC_PUBLISHED["accuracy_sd"]
    ax.axvspan(pub - sd, pub + sd, color=ORANGE, alpha=0.15, linewidth=0)
    ax.axvline(pub, color=ORANGE, linewidth=1)
    ax.text(pub, len(entries) - 0.35, "TF-C paper ± SD", fontsize=6, color=INK, ha="center")
    for i, (label, acc, _, _) in enumerate(reversed(entries)):
        ax.scatter(acc, np.full(len(acc), i), color=BLUE, s=14, zorder=3, edgecolors="white", linewidths=0.8)
        ax.plot([acc.mean()] * 2, [i - 0.3, i + 0.3], color=INK, linewidth=1.5)
    ax.set_yticks(range(len(entries)), [e[0].replace("$|$", "|").replace("$", "") for e in reversed(entries)],
                  fontsize=7)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(-0.6, len(entries) - 0.1)
    ax.set_xlabel("FD-B test accuracy (dot = seed, bar = mean)", fontsize=7)
    style(ax)
    save(fig, "fdb_overview")


# ---------------------------------------------------------------------------
# Herning band check
# ---------------------------------------------------------------------------
def herning():
    run = RUNS["band_check"]
    summ = read_csv(run, "summary.csv")
    drift = read_csv(run, "drift_control.csv")
    sweeps = read_csv(run, "sweeps.csv")
    d = summ.merge(drift, on=["channel", "band"])
    d["f_c"] = np.sqrt(np.maximum(d.f_lo, 1.0) * d.f_hi)  # geometric band centre (low band: 1 Hz lower edge)

    roles = sweeps.role.value_counts()
    for role, key in (("analysed", "Analysed"), ("noise_reference", "Noise"), ("non_rotating_0", "Zero"),
                      ("excluded", "Excluded")):
        number(f"Band{key}", f"{int(roles.get(role, 0)):,}")
    number("BandSweeps", f"{len(sweeps):,}")
    an = sweeps[sweeps.role == "analysed"]
    number("BandKappaMin", f"{an.kappa.min():.3f}")
    number("BandKappaMax", f"{an.kappa.max():.3f}")

    # Numbers quoted in the prose about the band check.
    ae, us = d[d.channel == "AE"], d[d.channel == "UL"]
    peaks = ae[ae.f_hi < 50e3].nlargest(1, "ratio_db").iloc[0], ae[ae.f_lo >= 50e3].nlargest(1, "ratio_db").iloc[0]
    number("BandAePeakLow", f"{peaks[0].ratio_db:.1f}~dB at {peaks[0].band}")
    number("BandAePeakHigh", f"{peaks[1].ratio_db:.1f}~dB at {peaks[1].band}")
    ae_core = ae[(ae.f_lo >= 11e3) & (ae.f_hi <= 450e3)]  # bands 12.6 ... 398 kHz
    number("BandAeCoreRpm", f"{ae_core.within_block_rpm_median.min():.2f}--{ae_core.within_block_rpm_median.max():.2f}")
    ae_top = ae[ae.f_lo >= 560e3]  # bands from 631 kHz up
    number("BandAeTopDb", f"${ae_top.ratio_db.max():.2f}$")
    number("BandAeTopRpm", f"${ae_top.within_block_rpm_median.min():.2f}$ to ${ae_top.within_block_rpm_median.max():.2f}$")
    us_mid = us[(us.f_lo >= 35e3) & (us.f_hi <= 225e3)]  # bands 39.8 ... 199.5 kHz
    number("BandUsMidDb", f"{us_mid.ratio_db.min():.1f}--{us_mid.ratio_db.max():.1f}")
    number("BandUsMidRpm", f"{us_mid.within_block_rpm_median.min():.2f}--{us_mid.within_block_rpm_median.max():.2f}")
    number("BandUsMidKappa", f"{us_mid.spearman_kappa.min():.2f}--{us_mid.spearman_kappa.max():.2f}")
    us_low = us[us.f_hi <= 7.1e3]  # bands up to 6.3 kHz
    number("BandUsLowDb", f"{us_low.ratio_db.min():.0f}--{us_low.ratio_db.max():.0f}")

    # Figure: excess power over the non-rotating reference, and Spearman correlations, per band.
    fig, axes = plt.subplots(2, 2, figsize=(6.3, 4.2), sharex=True)
    for col, (ch, title, band, nyq) in enumerate((("AE", "Acoustic emission (AE)", AE_BAND_HZ, AE_NYQUIST_HZ),
                                                  ("UL", "Ultrasound (US)", US_BAND_HZ, US_NYQUIST_HZ))):
        g = d[d.channel == ch]
        top, bot = axes[0, col], axes[1, col]
        for ax in (top, bot):
            ax.axvspan(*band, color=AQUA, alpha=0.12, linewidth=0)
            ax.axvline(nyq, color=MUTED, linestyle=":", linewidth=1)
            ax.set_xscale("log")
            style(ax)
        top.plot(g.f_c, g.ratio_db, color=INK, marker="o", markersize=3, linewidth=1.5)
        top.axhline(0, color=MUTED, linewidth=0.8)
        top.set_title(title, fontsize=8, color=INK, loc="left")
        bot.plot(g.f_c, g.within_block_rpm_median, color=BLUE, marker="o", markersize=3, linewidth=1.5,
                 label="rpm, within temperature blocks (median)")
        bot.plot(g.f_c, g.spearman_kappa, color=ORANGE, marker="s", markersize=3, linewidth=1.5,
                 label="κ, all analysed sweeps")
        bot.axhline(0, color=MUTED, linewidth=0.8)
        bot.set_ylim(-1, 1)
        bot.set_xlabel("Band centre (Hz), third-octave bands", fontsize=7)
    axes[0, 0].set_ylabel("Power above non-rotating\nreference (dB)", fontsize=7)
    axes[1, 0].set_ylabel("Spearman correlation\nof band power", fontsize=7)
    axes[1, 1].legend(fontsize=6, frameon=False, loc="lower left")
    # Shading (feature bands, D60/D61) and dotted lines (Nyquist after downsampling, D62)
    # are explained in the figure caption in results.tex.
    fig.tight_layout()
    save(fig, "herning_band_check")

    # Figure: the run's operating points over time, by role.
    colours = {"analysed": BLUE, "noise_reference": ORANGE, "non_rotating_0": AQUA, "excluded": MUTED}
    labels = {"analysed": "analysed", "noise_reference": "noise reference (rpm target 100)",
              "non_rotating_0": "rpm target 0", "excluded": "excluded (cool-down)"}
    fig, axes = plt.subplots(3, 1, figsize=(6.3, 4.0), sharex=True)
    for role, g in sweeps.groupby("role"):
        for ax, var in zip(axes, ("rpm", "temperature_c", "kappa")):
            ax.scatter(g.order, g[var], s=2, color=colours[role], linewidths=0, label=labels[role])
    for ax, lab in zip(axes, ("rpm (from VFD command)", "Temperature (°C)", "κ")):
        ax.set_ylabel(lab, fontsize=7)
        style(ax)
    axes[2].set_xlabel("Sweep (acquisition order)", fontsize=7)
    axes[0].legend(fontsize=6, frameon=False, markerscale=4, ncol=4, loc="upper left", bbox_to_anchor=(0, 1.3))
    save(fig, "herning_operating_points")

    # Appendix table: every band.
    HEAD = "Sensor & Band & Ratio (dB) & $\\rho$ rpm & $\\rho$ rpm in block & $\\rho$ temp. & $\\rho$ $\\kappa$ \\\\"
    lines = ["% Generated by scripts/make_report.py - do not edit by hand.",
             "\\begin{small}", "\\begin{longtable}{llrrrrr}",
             "\\caption{Herning band check, all bands. Ratio: median band power of analysed sweeps "
             "over that of the non-rotating reference. $\\rho$: Spearman correlation of band power with "
             "rpm (all analysed sweeps, and median within temperature blocks), temperature and $\\kappa$.}"
             "\\label{tab:bands}\\\\", "\\toprule", HEAD, "\\midrule", "\\endfirsthead",
             "\\toprule", HEAD, "\\midrule", "\\endhead"]  # header repeated on later pages, caption not
    for r in d.itertuples():
        values = [r.ratio_db, r.spearman_rpm, r.within_block_rpm_median, r.spearman_temperature_c, r.spearman_kappa]
        cells = [f"${v:.1f}$" if i == 0 else f"${v:.2f}$" for i, v in enumerate(values)]  # math mode: real minus signs
        lines.append(f"{'US' if r.channel == 'UL' else r.channel} & {r.band} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{longtable}", "\\end{small}"]
    with open(os.path.join(OUT, "tables", "herning_bands.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Cross-cutting tables
# ---------------------------------------------------------------------------
def reproduction_table(checks):
    """One row per reproduction check, with the D26 criterion as implemented."""
    rows = []
    for key, (method, dataset, published, acc, section) in checks.items():
        if acc is None:
            rows.append([section, method, dataset, f"{published:.4f}", "no test result", "--", "--", "--"])
            continue
        r = reproduction(acc, published)
        verdict = "pass" if r["pass"] else "fail"
        rows.append([section, method, dataset, f"{published:.4f}", f"{pm(r['mean'], r['sd'])} ($n={r['n']}$)",
                     f3(r["diff"]), f3(r["allowed"]), verdict])
        number(f"{key}Verdict", verdict)
        number(f"{key}Acc", pm(r["mean"], r["sd"]))
    write_table("reproduction", ["Section", "Method", "Data", "Published", "Ours (mean $\\pm$ SD)",
                                 "$|\\Delta|$", "Allowed", "Criterion"], rows, "lllccccc",
                "Reproduction checks: published accuracy against our mean over seeds, with the method's "
                "own probe and all labels. Criterion as implemented in \\texttt{scripts/evaluate.py} and "
                "\\texttt{scripts/tfc\\_official.py} (D26, D38): pass if "
                "$|\\Delta| \\le \\max(2\\,\\mathrm{SD}, 0.01)$.", "tab:repro")


def run_index_table():
    """Every folder in runs/ (D98): what it was for, its status, commit and device.
    Folders not in RUN_INDEX are listed as 'not indexed' and reported on the console."""
    folders = sorted(d for d in os.listdir(path("runs")) if os.path.isdir(path("runs", d)))
    HEAD = "Run folder (\\texttt{runs/}) & What & Status & Commit & Device \\\\"
    lines = ["% Generated by scripts/make_report.py - do not edit by hand.",
             "\\begin{footnotesize}", "\\begin{longtable}{>{\\raggedright\\arraybackslash}p{0.28\\linewidth}"
             ">{\\raggedright\\arraybackslash}p{0.22\\linewidth}>{\\raggedright\\arraybackslash}p{0.22\\linewidth}ll}",
             "\\caption{Run index: every folder in \\texttt{runs/} (D98). Commit with $^{*}$: the working "
             "tree had uncommitted changes when the run started, so the commit alone does not reproduce "
             "the code.}\\label{tab:runs}\\\\",
             "\\toprule", HEAD, "\\midrule", "\\endfirsthead", "\\toprule", HEAD, "\\midrule", "\\endhead"]
    missing = []
    for d in folders:
        what, status = RUN_INDEX.get(d, ("--", "Not indexed"))
        if d not in RUN_INDEX:
            missing.append(d)
        try:
            git = read_json(f"runs/{d}", "git.json")
            commit = (git["commit"][:7] if git.get("commit") else "none") + ("$^{*}$" if git.get("uncommitted_changes") else "")
        except FileNotFoundError:
            commit = "--"
        try:
            device = str(read_yaml(f"runs/{d}").get("device", "--"))
        except FileNotFoundError:
            device = "--"
        status = status if status.startswith("In ") else tex(status).replace("\\_", "\\_\\allowbreak{}")
        name = tex(d).replace("\\_", "\\_\\allowbreak{}")  # long folder names may break after an underscore
        lines.append(f"\\texttt{{{name}}} & {tex(what)} & {status} & {commit} & {device} \\\\")
    lines += ["\\bottomrule", "\\end{longtable}", "\\end{footnotesize}"]
    with open(os.path.join(OUT, "tables", "run_index.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    for name in RUN_INDEX:
        if name not in folders:
            print(f"Warning: RUN_INDEX names {name}, which is not in runs/")
    for d in missing:
        print(f"Warning: runs/{d} is not in RUN_INDEX (listed as 'Not indexed')")


def main():
    os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)

    checks = forda()
    full = tfc_official_full()
    ours = tfc_ours()
    subset = tfc_official_subset()
    tfc_losses_figure(full)
    fdb_overview(subset, ours)
    herning()

    # Reproduction checks; the key is also the prefix of the quoted numbers (\TsvecVerdict, ...).
    pub = TFC_PUBLISHED["accuracy"]
    rows = {"Tsvec": checks["TS2Vec"], "Tloss": checks["T-Loss"], "Minirocket": checks["MiniRocket"],
            "Tsjepa": checks["TS-JEPA, official masking"],
            "TfcFull": ("TF-C, official, full data", "FD", pub, None, "\\ref{tfc-full}"),
            "TfcOurs": ("TF-C, ours", "FD", pub, ours.accuracy, "\\ref{tfc-ours}"),
            "TfcSubset": ("TF-C, official, subset", "FD", pub, subset[subset.protocol == "official"].accuracy,
                          "\\ref{tfc-subset}")}
    reproduction_table(rows)
    run_index_table()

    with open(os.path.join(OUT, "tables", "numbers.tex"), "w", encoding="utf-8") as f:
        f.write("% Generated by scripts/make_report.py - do not edit by hand.\n")
        for k, v in sorted(NUMBERS.items()):
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print(f"Wrote {len(os.listdir(os.path.join(OUT, 'tables')))} tables and "
          f"{len(os.listdir(os.path.join(OUT, 'figures')))} figures into {OUT}/")


if __name__ == "__main__":
    if len(sys.argv) > 1:  # optional: repository root (default: current folder)
        ROOT = sys.argv[1]
        OUT = os.path.join(ROOT, "report")
    main()
