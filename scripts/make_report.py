"""Tables, figures and cited numbers for the results document (report/results.tex).

Usage (from the repository root):
    uv run python scripts/make_report.py

Reads the run folders listed in RUNS (runs/ is not committed) and writes, into report/
(committed):
    tables/*.tex     LaTeX tables, included with \\input
    tables/numbers.tex  \\newcommand definitions for every number quoted in the prose
    figures/*.pdf    figures, included with \\includegraphics
Every number in the document comes from here; none is typed by hand. Rerun after a new
run has been added to RUNS and described in results.tex.

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
# Which run folder belongs to which result entry (R-number in results.tex).
# ---------------------------------------------------------------------------
RUNS = {
    "R2": "runs/20261007_214220_ts2vec_forda_cpu_check",
    "R3": "runs/20261008_084722_ts2vec_forda",
    "R4": "runs/20261008_092321_tloss_forda",
    "R5a": "runs/20261008_134951_tfc_official_fd",  # first attempt, seed 0 only
    "R5b": "runs/20261008_143021_tfc_official_fd",  # second attempt, all seeds
    "R5c": "runs/20261008_155516_tfc_official_fd",  # third attempt, after the D44 change
    "R6": "runs/20261008_142751_tfc_fd",
    "R7": "runs/20261008_151014_probe_tfc_fd",
    "R8": "runs/20261008_162447_probe_spectrum_fd",
    "R9": "runs/20261008_171159_tfc_official_fd_subset",
    "R10": "runs/20261009_085307_band_check_herning",
}
# Run folders with no results (started and stopped before writing anything but the config).
EMPTY_RUNS = {
    "R6": "runs/20261008_143011_tfc_fd",
    "R10": "runs/20261009_085150_band_check_herning",
}

# Published numbers that are not in the run configs: standard deviations of the TF-C paper
# (arXiv 2206.08496, Table 5, FD-A -> FD-B, 5 runs), as recorded in D38.
TFC_PUBLISHED = {"accuracy": 0.8934, "accuracy_sd": 0.0379, "macro_f1": 0.9162, "macro_f1_sd": 0.0826}

# Herning feature bands and downsampled Nyquist frequencies, as decided (for shading in the figure).
AE_BAND_HZ = (50e3, 560e3)  # D61
US_BAND_HZ = (1e3 * 10 ** -0.15, 126e3 * 10 ** 0.15)  # D60: base-10 octaves, centres 1.0 ... 126 kHz
AE_NYQUIST_HZ = 1.25e6 / 2  # D62: AE downsampled by 2
US_NYQUIST_HZ = 0.5e6 / 2  # D62: US downsampled by 5

# Colours (light-mode categorical slots 1-3 of the validated reference palette) and ink.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#8a8984", "#e4e3df"

# LaTeX command names may only contain letters: R3 -> \\RthreeVerdict etc.
WORD = {"R2": "Rtwo", "R3": "Rthree", "R4": "Rfour", "R5": "Rfive", "R6": "Rsix", "R9": "Rnine"}

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


def read_git(run):
    with open(path(run, "git.json")) as f:
        return json.load(f)


def f3(x):
    """Three decimals, or an en dash for a missing value."""
    return "--" if pd.isna(x) else f"{x:.3f}"


def pm(mean, sd):
    """'mean ± sd' in math mode, three decimals."""
    return f"${mean:.3f} \\pm {sd:.3f}$"


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
             f"\\begin{{tabular}}{{{colspec}}}", "\\toprule",
             " & ".join(header) + " \\\\", "\\midrule"]
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
# FordA: TS2Vec and T-Loss (R2, R3, R4)
# ---------------------------------------------------------------------------
def forda():
    methods = {"TS2Vec": RUNS["R3"], "T-Loss": RUNS["R4"]}
    probes = {"svm_ts2vec": "SVM (TS2Vec protocol)", "logreg": "Logistic regression"}

    # Table: all probes and label budgets, mean ± SD over the 3 seeds.
    rows = []
    for method, run in methods.items():
        res = read_csv(run, "results.csv")
        for probe, probe_name in probes.items():
            for frac, g in res[res.probe == probe].groupby("label_fraction"):
                rows.append([method, probe_name, f"{frac:.0%}".replace("%", "\\%"), str(g.n_labels.iloc[0]),
                             pm(g.accuracy.mean(), g.accuracy.std()), pm(g.macro_f1.mean(), g.macro_f1.std())])
        rows.append("MIDRULE")
    write_table("forda_probes", ["Method", "Probe", "Labels", "$n$", "Accuracy", "Macro-F1"], rows[:-1],
                "llrrcc", "FordA test set, frozen representations: mean $\\pm$ SD over 3 seeds "
                "(R3, R4). The label subset is redrawn for each seed (D29).", "tab:forda")

    # Figure: accuracy vs label budget, one panel per probe, published values as reference lines.
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.5), sharey=True)
    for ax, (probe, probe_name) in zip(axes, probes.items()):
        for (method, run), colour, dx in zip(methods.items(), (BLUE, ORANGE), (-0.03, 0.03)):
            res = read_csv(run, "results.csv")
            g = res[res.probe == probe].groupby("label_fraction").accuracy
            x = np.log10(g.mean().index * 100)
            ax.errorbar(x + dx, g.mean(), yerr=g.std(), color=colour, linewidth=2, marker="o",
                        markersize=4, capsize=2, label=method)
            seeds = res[res.probe == probe]
            ax.scatter(np.log10(seeds.label_fraction * 100) + dx, seeds.accuracy, s=6, color=colour,
                       alpha=0.5, linewidths=0)
            if probe == "svm_ts2vec":  # the published numbers use this protocol
                pub = read_yaml(run)["evaluation"]["published_accuracy"]
                ax.axhline(pub, color=colour, linestyle="--", linewidth=1)
                ax.text(-0.05, pub + 0.003, f"{method} published {pub:.3f}", fontsize=6, color=INK)
        ax.set_xticks([0, 1, 2], ["1 %", "10 %", "100 %"])
        ax.set_xlim(-0.2, 2.2)
        ax.set_title(probe_name, fontsize=8, color=INK, loc="left")
        ax.set_xlabel("Training labels used by the probe", fontsize=7)
        style(ax)
    axes[0].set_ylabel("Test accuracy", fontsize=7)
    axes[1].legend(fontsize=7, frameon=False, loc="lower right")
    save(fig, "forda_label_budget")

    # Reproduction checks (SVM probe, 100 % labels) and the preliminary CPU check (R2).
    out = {}
    for key, method in (("R2", "TS2Vec (CPU check)"), ("R3", "TS2Vec"), ("R4", "T-Loss")):
        run = RUNS[key]
        res = read_csv(run, "results.csv")
        acc = res[(res.probe == "svm_ts2vec") & (res.label_fraction == 1.0)].accuracy
        out[key] = (method, "FordA", read_yaml(run)["evaluation"]["published_accuracy"], acc)
    return out


# ---------------------------------------------------------------------------
# TF-C on FD-A -> FD-B (R5 - R9)
# ---------------------------------------------------------------------------
def pretrain_losses(run):
    """{seed: loss per epoch (NaN where not finite / not written)} for a TF-C run."""
    seeds = sorted(d for d in os.listdir(path(run)) if d.startswith("seed_"))
    return {int(s.split("_")[1]): read_csv(run, f"{s}/pretrain_loss.csv") for s in seeds}


def tfc_official_full():
    """R5: official TF-C on the full data. Per seed: how far pretraining got, and whether
    fine-tuning wrote anything."""
    losses = pretrain_losses(RUNS["R5c"])
    # The second and third attempts ran the same settings; check the loss curves are identical.
    other = pretrain_losses(RUNS["R5b"])
    identical = all(losses[s].loss.equals(other[s].loss) for s in losses)
    first = pretrain_losses(RUNS["R5a"])[0]
    # The first attempt (older code) kept writing after the loss became non-finite: compare finite epochs.
    identical_first = first.loss.dropna().reset_index(drop=True).equals(losses[0].loss.dropna().reset_index(drop=True))
    number("RfiveIdentical", "identical" if identical else "not identical")
    number("RfiveFirstIdentical", "identical" if identical_first else "not identical")

    rows = []
    for seed, df in losses.items():
        finite = df.loss.notna()
        n_ok = int(finite.sum())
        first_bad = "--" if n_ok == len(df) else str(int(df.epoch[~finite].iloc[0]))
        finetuned = os.path.exists(path(RUNS["R5c"], f"seed_{seed}", "epochs.csv"))
        rows.append([str(seed), str(len(df)), str(n_ok), first_bad, f3(df.loss[finite].iloc[-1]),
                     "yes" if finetuned else "no"])
    n_div = sum(r[3] != "--" for r in rows)
    number("RfiveDiverged", str(n_div))
    number("RfiveSeeds", str(len(rows)))
    number("RfiveResultsWritten", "yes" if os.path.exists(path(RUNS["R5c"], "results.csv")) else "no")
    write_table("tfc_official_full", ["Seed", "Epochs run", "Finite", "First non-finite", "Last finite loss",
                                      "Fine-tuning output"], rows, "rccccc",
                "Official TF-C, full FD-A pretraining data (R5, third attempt). Pretraining loss "
                "per epoch is empty in \\texttt{pretrain\\_loss.csv} once it is no longer finite.",
                "tab:tfc-official-full",
                notes=f"Second and third attempt: pretraining losses {'identical' if identical else 'NOT identical'} "
                      "for every seed.")
    return losses


def tfc_official_subset():
    """R9: official TF-C with the debugging subset."""
    run = RUNS["R9"]
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
                "Official TF-C with the debugging subset (R9). Official protocol: epoch with the best "
                "test accuracy; clean protocol: epoch with the best validation macro-F1 (D43). Last "
                "column: macro recall on test and validation is exactly $1/3$ at every one of the "
                "40 fine-tuning epochs.", "tab:tfc-official-subset")
    acc = res[res.protocol == "official"].accuracy
    number("RnineAcc", pm(acc.mean(), acc.std()))
    f1 = res[res.protocol == "official"].macro_f1
    number("RnineFone", pm(f1.mean(), f1.std()))
    return res


def tfc_ours():
    """R6: our TF-C."""
    run = RUNS["R6"]
    res = read_csv(run, "results.csv")
    cols = ["accuracy", "precision", "recall", "f1", "auroc", "auprc"]
    rows = [[str(r.seed), str(int(r.epoch))] + [f3(getattr(r, c)) for c in cols] for r in res.itertuples()]
    rows += ["MIDRULE", ["Mean $\\pm$ SD", ""] + [pm(res[c].mean(), res[c].std()) for c in cols]]
    write_table("tfc_ours", ["Seed", "Epoch", "Acc.", "Prec.", "Recall", "F1", "AUROC", "AUPRC"], rows,
                "rrcccccc", "Our TF-C (D41, D42), FD-A $\\rightarrow$ FD-B test set (R6). Epoch: "
                "fine-tuning epoch with the best validation macro-F1; the test set is scored once, "
                "for that epoch. Precision, recall and F1 are macro averages.", "tab:tfc-ours")
    number("RsixAcc", pm(res.accuracy.mean(), res.accuracy.std()))
    number("RsixFone", pm(res.f1.mean(), res.f1.std()))
    number("RsixAccMin", f3(res.accuracy.min()))
    number("RsixAccMax", f3(res.accuracy.max()))

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
    number("RsixNval", str(n_val))
    return res


def tfc_losses_figure(full):
    """Pretraining loss per epoch: official full data (R5), official subset (R9), ours (R6)."""
    panels = [("Official TF-C, full data (R5)", full),
              ("Official TF-C, debugging subset (R9)", pretrain_losses(RUNS["R9"])),
              ("Our TF-C (R6)", pretrain_losses(RUNS["R6"]))]
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
    """R7, R8 and the TF-C results side by side on the FD-B test set."""
    probes = read_csv(RUNS["R7"], "results.csv")
    spectrum = read_csv(RUNS["R8"], "results.csv")
    name = {"svm_ts2vec": "SVM", "logreg": "log. reg."}
    entries = []  # (label, accuracies per seed, macro-F1 per seed, entry)
    official = subset_res[subset_res.protocol == "official"]
    entries.append(("Official TF-C, subset, fine-tuned", official.accuracy, official.macro_f1, "R9"))
    entries.append(("Our TF-C, fine-tuned", ours_res.accuracy, ours_res.f1, "R6"))
    for enc, enc_name in (("pretrained", "pretrained TF-C encoder"), ("random_init", "untrained encoder")):
        for probe in ("svm_ts2vec", "logreg"):
            g = probes[(probes.encoder == enc) & (probes.probe == probe)]
            entries.append((f"{name[probe]} on {enc_name}", g.accuracy, g.macro_f1, "R7"))
    for probe in ("svm_ts2vec", "logreg"):
        g = spectrum[spectrum.probe == probe]
        entries.append((f"{name[probe]} on $|$FFT$|$ spectrum", g.accuracy, g.macro_f1, "R8"))

    rows = [["TF-C paper (published)", "--", pm(TFC_PUBLISHED["accuracy"], TFC_PUBLISHED["accuracy_sd"]),
             pm(TFC_PUBLISHED["macro_f1"], TFC_PUBLISHED["macro_f1_sd"]), "5"], "MIDRULE"]
    for label, acc, f1, entry in entries:
        cell = (lambda s: f3(s.iloc[0])) if len(acc) == 1 else (lambda s: pm(s.mean(), s.std()))
        rows.append([label, entry, cell(acc), cell(f1), str(len(acc))])
    write_table("fdb_overview", ["Model / probe", "Entry", "Accuracy", "Macro-F1", "Seeds"], rows, "llccr",
                "FD-B test set: TF-C results and the diagnostics (R6--R9). Probes use all 60 FD-B "
                "training labels (D45, D46). The $|$FFT$|$ probes are deterministic (one value).",
                "tab:fdb")
    for enc, key in (("pretrained", "Pre"), ("random_init", "Rand")):
        for probe, pkey in (("svm_ts2vec", "Svm"), ("logreg", "Logreg")):
            g = probes[(probes.encoder == enc) & (probes.probe == probe)].accuracy
            number(f"Rseven{key}{pkey}", pm(g.mean(), g.std()))
    for probe, pkey in (("svm_ts2vec", "Svm"), ("logreg", "Logreg")):
        number(f"Reight{pkey}", f3(spectrum[spectrum.probe == probe].accuracy.iloc[0]))

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
# Herning band check (R10)
# ---------------------------------------------------------------------------
def herning():
    run = RUNS["R10"]
    summ = read_csv(run, "summary.csv")
    drift = read_csv(run, "drift_control.csv")
    sweeps = read_csv(run, "sweeps.csv")
    d = summ.merge(drift, on=["channel", "band"])
    d["f_c"] = np.sqrt(np.maximum(d.f_lo, 1.0) * d.f_hi)  # geometric band centre (low band: 1 Hz lower edge)

    roles = sweeps.role.value_counts()
    for role, key in (("analysed", "Analysed"), ("noise_reference", "Noise"), ("non_rotating_0", "Zero"),
                      ("excluded", "Excluded")):
        number(f"Rten{key}", f"{int(roles.get(role, 0)):,}")
    number("RtenSweeps", f"{len(sweeps):,}")
    an = sweeps[sweeps.role == "analysed"]
    number("RtenKappaMin", f"{an.kappa.min():.3f}")
    number("RtenKappaMax", f"{an.kappa.max():.3f}")

    # Numbers quoted in the prose about the band check.
    ae, us = d[d.channel == "AE"], d[d.channel == "UL"]
    peaks = ae[ae.f_hi < 50e3].nlargest(1, "ratio_db").iloc[0], ae[ae.f_lo >= 50e3].nlargest(1, "ratio_db").iloc[0]
    number("RtenAePeakLow", f"{peaks[0].ratio_db:.1f}~dB at {peaks[0].band}")
    number("RtenAePeakHigh", f"{peaks[1].ratio_db:.1f}~dB at {peaks[1].band}")
    ae_core = ae[(ae.f_lo >= 11e3) & (ae.f_hi <= 450e3)]  # bands 12.6 ... 398 kHz
    number("RtenAeCoreRpm", f"{ae_core.within_block_rpm_median.min():.2f}--{ae_core.within_block_rpm_median.max():.2f}")
    ae_top = ae[ae.f_lo >= 560e3]  # bands from 631 kHz up
    number("RtenAeTopDb", f"${ae_top.ratio_db.max():.2f}$")
    number("RtenAeTopRpm", f"${ae_top.within_block_rpm_median.min():.2f}$ to ${ae_top.within_block_rpm_median.max():.2f}$")
    us_mid = us[(us.f_lo >= 35e3) & (us.f_hi <= 225e3)]  # bands 39.8 ... 199.5 kHz
    number("RtenUsMidDb", f"{us_mid.ratio_db.min():.1f}--{us_mid.ratio_db.max():.1f}")
    number("RtenUsMidRpm", f"{us_mid.within_block_rpm_median.min():.2f}--{us_mid.within_block_rpm_median.max():.2f}")
    number("RtenUsMidKappa", f"{us_mid.spearman_kappa.min():.2f}--{us_mid.spearman_kappa.max():.2f}")
    us_low = us[us.f_hi <= 7.1e3]  # bands up to 6.3 kHz
    number("RtenUsLowDb", f"{us_low.ratio_db.min():.0f}--{us_low.ratio_db.max():.0f}")

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
             "\\caption{Herning band check, all bands (R10). Ratio: median band power of analysed sweeps "
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
    for key, (method, dataset, published, acc) in checks.items():
        if acc is None:
            rows.append([key, method, dataset, f"{published:.4f}", "no test result", "--", "--", "--"])
            continue
        r = reproduction(acc, published)
        verdict = "pass" if r["pass"] else "fail"
        rows.append([key, method, dataset, f"{published:.4f}", f"{pm(r['mean'], r['sd'])} ($n={r['n']}$)",
                     f3(r["diff"]), f3(r["allowed"]), verdict])
        number(f"{WORD[key]}Verdict", verdict)
    write_table("reproduction", ["Entry", "Method", "Data", "Published", "Ours (mean $\\pm$ SD)",
                                 "$|\\Delta|$", "Allowed", "Criterion"], rows, "lllccccc",
                "Reproduction checks: published accuracy against our mean over seeds. Criterion as "
                "implemented in \\texttt{scripts/evaluate.py} and \\texttt{scripts/tfc\\_official.py} "
                "(D26, D38): pass if $|\\Delta| \\le \\max(2\\,\\mathrm{SD}, 0.01)$.", "tab:repro")
    for key in ("R3", "R4"):  # accuracies quoted in the prose (R6, R9 are set in their own functions)
        acc = checks[key][3]
        number(f"{WORD[key]}Acc", pm(acc.mean(), acc.std()))


def provenance_table():
    """Run folder, commit and device for every run behind the document."""
    rows = []
    for key, run in list(RUNS.items()) + [(k + " (empty)", v) for k, v in EMPTY_RUNS.items()]:
        git = read_git(run)
        commit = git["commit"][:7] if git.get("commit") else "none"
        dirty = {True: "yes", False: "no", None: "--"}[git.get("uncommitted_changes")]
        try:
            device = read_yaml(run).get("device", "--")
        except FileNotFoundError:
            device = "--"
        rows.append([key.replace("R5a", "R5").replace("R5b", "R5").replace("R5c", "R5"),
                     "\\texttt{" + os.path.basename(run).replace("_", "\\_") + "}", commit, dirty, str(device)])
    write_table("provenance", ["Entry", "Run folder (\\texttt{runs/})", "Commit", "Uncommitted", "Device"], rows,
                "llccc", "Run folders behind each entry. 'Uncommitted': the working tree had changes "
                "when the run started, so the commit alone does not reproduce the code.", "tab:provenance")


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

    pub = TFC_PUBLISHED["accuracy"]
    checks["R5"] = ("TF-C, official, full data", "FD", pub, None)
    checks["R6"] = ("TF-C, ours", "FD", pub, ours.accuracy)
    checks["R9"] = ("TF-C, official, subset", "FD", pub, subset[subset.protocol == "official"].accuracy)
    reproduction_table(checks)
    provenance_table()

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
