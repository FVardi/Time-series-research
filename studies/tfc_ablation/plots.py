"""TF-C study: figure functions for the ablation stages A-E (loss, collapse measures, spectra, probes, PCA).

Used by the notebook figures_forda.ipynb in this folder, which shows the figures inline.
Nothing is written to disk: the figures exist only in the notebook, and the notebook is
committed with outputs cleared. What each figure shows and how to read it: the notebook's
markdown cells and README.md, section 'Figures'.

Only measured values are drawn; nothing is fitted or smoothed. With several seeds a line is
the mean over seeds and the shaded band runs from the smallest to the largest seed.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Colours and style (same palette and axes style as scripts/make_report.py).
# A and B never share a panel (A has only h_T, B only h_F), so both use slot 1:
# "one branch alone". E, the full TF-C and the reference point, is drawn in ink.
# Blue, aqua and yellow pass the palette check for all pairs; every line is also
# labelled directly at its end, so identity never rests on colour alone.
# The variants F-H (D101) take the colour of the stage they modify and a line style:
# F (S_TF only, with E's weight) and H (augmented pairs as positives) are variants of D,
# yellow dashed and dotted; G (hinged L_C) is a variant of E, ink dashed.
# ---------------------------------------------------------------------------
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#0b0b0b", "#8a8984", "#e4e3df"
STAGES = {  # stage -> (colour, line style, legend text)
    "A": (BLUE, "-", "A, B  one branch alone (A: time, L_T; B: frequency, L_F)"),
    "B": (BLUE, "-", None),  # shares the legend entry of A
    "C": (AQUA, "-", "C  both branches, L_T + L_F"),
    "D": (YELLOW, "-", "D  λ(L_T + L_F) + (1 − λ) S_TF"),
    "E": (INK, "-", "E  full TF-C, λ(L_T + L_F) + (1 − λ) L_C"),
    "F": (YELLOW, "--", "F  λ(L_T + L_F) + 3(1 − λ) S_TF (E without pair terms)"),
    "G": (INK, "--", "G  hinged L_C: Σ max(0, S_TF − S_pair + δ)"),
    "H": (YELLOW, ":", "H  pairs as positives: (S_TF + ΣS_pair) / 4"),
}
STAGE_TEXT = {st: STAGES[st][2] for st in STAGES}  # titles of per-stage figures
STAGE_TEXT.update({"A": "A  time branch alone, L_T", "B": "B  frequency branch alone, L_F"})
CLASS_COLOURS = (BLUE, ORANGE)  # PCA scatter: the two FordA classes

# Representations in a fixed grid position, so the same panel means the same thing in every figure.
REPS = ["h_t", "h_f", "h", "z_t", "z_f", "z"]
REP_TITLE = {"h_t": "h_T (time encoder)", "h_f": "h_F (frequency encoder)", "h": "h = [h_T; h_F]",
             "z_t": "z_T (time projector)", "z_f": "z_F (frequency projector)", "z": "z = [z_T; z_F]"}

# Loss parts as logged by ablation.py; which stages log which part follows from the stage.
LOSS_PARTS = {"loss_t": "L_T (time, NT-Xent on h_T)", "loss_f": "L_F (frequency, NT-Xent on h_F)",
              "s_tf": "S_TF (NT-Xent between z_T and z_F)", "loss_c": "L_C (consistency loss)"}


def style(ax):
    """Recessive axes and grid (as scripts/make_report.py)."""
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=7)


def show(fig):
    """Show the figure in the notebook (and close it, so it is not shown twice)."""
    plt.show()
    plt.close(fig)


def spread(ax, items, min_gap=0.075):
    """[(x, y, text)] in data coordinates -> [(x, y_axes, text)] with the y positions (axes fractions)
    moved apart so labels do not overlap."""
    ax.relim()
    ax.autoscale_view()  # the axis limits must be final before data are converted to axes fractions
    to_axes = ax.transData + ax.transAxes.inverted()
    pos = sorted((float(np.clip(to_axes.transform((x, y))[1], 0, 1)), x, text) for x, y, text in items)
    out, last = [], -np.inf
    for y, x, text in pos:
        last = max(y, last + min_gap)
        out.append((x, last, text))
    return out


def end_labels(ax, items):
    """Direct labels right of the panel, at the height of each line's last point: items = [(x, y, text)]."""
    if items:
        for _, y, text in spread(ax, items):
            ax.annotate(text, xy=(1.02, y), xycoords="axes fraction", fontsize=6.5, color=INK, va="center")


def point_labels(ax, items, dx=4, ha="left"):
    """Direct labels next to given points (x in data units), moved apart vertically: items = [(x, y, text)]."""
    if items:
        for x, y, text in spread(ax, items):
            ax.annotate(text, xy=(x, y), xycoords=("data", "axes fraction"), xytext=(dx, 0),
                        textcoords="offset points", fontsize=6.5, color=INK, ha=ha, va="center")


def stage_legend(fig, stages, y=1.0):
    """One legend for the whole figure, with an entry for each stage that appears."""
    handles, seen = [], set()
    for st in stages:
        colour, ls, text = STAGES[st]
        if st == "B" and "A" in stages:
            continue  # A and B share one entry
        if text is None:
            text = "B  frequency branch alone, L_F"
        if text not in seen:
            seen.add(text)
            handles.append(plt.Line2D([], [], color=colour, linestyle=ls, linewidth=1.5, label=text))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, y), ncol=2, fontsize=6.5, frameon=False)


# ---------------------------------------------------------------------------
# Reading the runs
# ---------------------------------------------------------------------------
def dirs_for_variance(eigenvalues, share):
    """Smallest number of directions (largest eigenvalues first) that hold the given share of the variance."""
    ev = np.sort(np.asarray(eigenvalues))[::-1]
    return int(np.searchsorted(np.cumsum(ev) / ev.sum(), share) + 1)


def load_run(folder, variance_share):
    """All seeds of one ablation run: collapse measures, losses, spectra and probes (if written yet)."""
    seeds = sorted(int(n.split("_")[1]) for n in os.listdir(folder) if n.startswith("seed_"))
    if not seeds:
        raise FileNotFoundError(f"{folder}: no seed_<k> folders")
    coll, loss, spec = [], [], []
    for s in seeds:
        d = os.path.join(folder, f"seed_{s}")
        coll.append(pd.read_csv(os.path.join(d, "collapse.csv")).assign(seed=s))
        loss.append(pd.read_csv(os.path.join(d, "pretrain_loss.csv")).assign(seed=s))
        spec.append(pd.read_csv(os.path.join(d, "spectra.csv")).assign(seed=s))
    coll, loss, spec = pd.concat(coll), pd.concat(loss), pd.concat(spec)
    # Number of directions holding variance_share of the variance, from the stored eigenvalues.
    dirs = (spec.groupby(["seed", "epoch", "representation"]).eigenvalue
            .apply(lambda e: dirs_for_variance(e, variance_share)).rename("dirs").reset_index())
    coll = coll.merge(dirs, on=["seed", "epoch", "representation"], how="left")
    probes_path = os.path.join(folder, "probes.csv")
    probes = pd.read_csv(probes_path) if os.path.exists(probes_path) else None
    if probes is None:
        print(f"  note: {folder} has no probes.csv yet; it is left out of the probe figure", flush=True)
    with open(os.path.join(folder, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    return {"folder": folder, "cfg": cfg, "seeds": seeds, "collapse": coll, "loss": loss, "spectra": spec,
            "probes": probes}


def over_seeds(df, x, y):
    """Mean, minimum and maximum of column y over seeds, per value of x."""
    g = df.groupby(x)[y]
    return g.mean(), g.min(), g.max()


def draw_line(ax, x, mean, lo, hi, st, n_seeds, **kw):
    """Line of stage st in its colour and line style; band from the smallest to the largest seed."""
    colour, ls, _ = STAGES[st]
    ax.plot(x, mean, color=colour, linestyle=ls, linewidth=1.3, **kw)
    if n_seeds > 1:
        ax.fill_between(x, lo, hi, color=colour, alpha=0.15, linewidth=0)


def grid(n_panels, ncols=3, height=2.0):
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.3, height * nrows), squeeze=False)
    return fig, axes.ravel()


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def nt_xent_bounds(runs):
    """NT-Xent at chance (all 2B - 1 candidates alike) and its maximum (partner opposite, all negatives aligned)."""
    B = runs_batch_size(runs)
    tau = {r["cfg"]["tfc"]["loss"]["temperature"] for r in runs.values()}
    if len(tau) != 1:
        raise ValueError(f"stages use different temperatures {tau}")
    tau = tau.pop()
    return np.log(2 * B - 1), np.log(1 + (2 * B - 2) * np.exp(2 / tau))


def with_mean_pair(loss, delta):
    """Adds the mean of the three S_pair terms, derived from the log: L_C = 3 S_TF - sum(S_pair) + 3 delta.
    Available for every run that logged L_C, also those from before the terms were logged separately."""
    if "loss_c" not in loss.columns:
        return loss
    return loss.assign(s_pair_mean=(3 * loss.s_tf + 3 * delta - loss.loss_c) / 3)


def loss_figure(runs):
    """F1: each loss part over epochs, stages overlaid; last panel: mean S_pair derived from L_C."""
    parts = {**LOSS_PARTS, "s_pair_mean": "mean of the three S_pair (derived from L_C)"}
    fig, axes = grid(len(parts), ncols=2, height=2.1)
    axes[-1].axis("off")  # 5 panels in a 3 x 2 grid
    chance, maximum = nt_xent_bounds(runs)
    for ax, (part, title) in zip(axes, parts.items()):
        labels = []
        for st, r in runs.items():
            loss = with_mean_pair(r["loss"], r["cfg"]["tfc"]["loss"]["delta"])
            if part not in loss.columns:
                continue
            m, lo, hi = over_seeds(loss, "epoch", part)
            draw_line(ax, m.index, m, lo, hi, st, len(r["seeds"]))
            labels.append((m.index[-1], m.iloc[-1], st))
        if part == "loss_c":
            delta = next(iter(runs.values()))["cfg"]["tfc"]["loss"]["delta"]
            ax.axhline(3 * delta, color=MUTED, linestyle=":", linewidth=1)
            ax.text(0.01, 3 * delta, " 3δ: S_TF = S_pair", transform=ax.get_yaxis_transform(), fontsize=6,
                    color=MUTED, va="bottom")
        else:
            ax.axhline(chance, color=MUTED, linestyle=":", linewidth=1)
            ax.text(0.01, chance, f" ln {2 * runs_batch_size(runs) - 1} (no discrimination)",
                    transform=ax.get_yaxis_transform(), fontsize=6, color=MUTED, va="bottom")
        if part == "s_pair_mean":
            ax.axhline(maximum, color=MUTED, linestyle=":", linewidth=1)
            ax.text(0.01, maximum, f" maximum {maximum:.1f}", transform=ax.get_yaxis_transform(), fontsize=6,
                    color=MUTED, va="top")
        ax.set_title(title, fontsize=8, color=INK, loc="left")
        ax.set_xlabel("Epoch", fontsize=7)
        style(ax)
        end_labels(ax, labels)
    stage_legend(fig, list(runs))
    fig.tight_layout(h_pad=1.5, w_pad=2.5)
    show(fig)


def runs_batch_size(runs):
    sizes = {r["cfg"]["tfc"]["pretrain"]["batch_size"] for r in runs.values()}
    if len(sizes) != 1:
        raise ValueError(f"stages use different batch sizes {sizes}; the chance line would differ per stage")
    return sizes.pop()


def pair_loss_figure(runs):
    """The three S_pair terms of L_C separately, for runs that logged them (D101); stages overlaid."""
    pairs = {"s_pair_1": "S_pair,1: z_T vs augmented z_F", "s_pair_2": "S_pair,2: augmented z_T vs z_F",
             "s_pair_3": "S_pair,3: augmented z_T vs augmented z_F"}
    logged = {st: r for st, r in runs.items() if "s_pair_1" in r["loss"].columns}
    if not logged:
        print("no run logged the S_pair terms separately (runs made before D101): figure skipped")
        return
    chance, maximum = nt_xent_bounds(runs)
    fig, axes = grid(3, ncols=3, height=2.2)
    for ax, (part, title) in zip(axes, pairs.items()):
        labels = []
        for st, r in logged.items():
            m, lo, hi = over_seeds(r["loss"], "epoch", part)
            draw_line(ax, m.index, m, lo, hi, st, len(r["seeds"]))
            labels.append((m.index[-1], m.iloc[-1], st))
        for y, text, va in ((chance, f" ln {2 * runs_batch_size(runs) - 1}", "bottom"), (maximum, " maximum", "top")):
            ax.axhline(y, color=MUTED, linestyle=":", linewidth=1)
            ax.text(0.01, y, text, transform=ax.get_yaxis_transform(), fontsize=6, color=MUTED, va=va)
        ax.set_title(title, fontsize=7.5, color=INK, loc="left")
        ax.set_xlabel("Epoch", fontsize=7)
        style(ax)
        end_labels(ax, labels)
    stage_legend(fig, list(logged))
    fig.tight_layout(w_pad=2.5)
    show(fig)


def measure_figure(runs, column, ylabel, log=False, note=None):
    """F2, F4, F5, F6: one collapse measure over epochs, one panel per representation, stages overlaid."""
    fig, axes = grid(len(REPS))
    for ax, rep in zip(axes, REPS):
        labels = []
        for st, r in runs.items():
            c = r["collapse"][r["collapse"].representation == rep]
            if c.empty:
                continue
            m, lo, hi = over_seeds(c, "epoch", column)
            draw_line(ax, m.index, m, lo, hi, st, len(r["seeds"]))
            labels.append((m.index[-1], m.iloc[-1], st))
        if not labels:
            ax.text(0.5, 0.5, "not measured\nin these stages", ha="center", va="center", fontsize=7,
                    color=MUTED, transform=ax.transAxes)
            ax.set_xticks([]), ax.set_yticks([])
            for side in ax.spines.values():
                side.set_visible(False)
        else:
            if log:
                ax.set_yscale("log")
            style(ax)
            end_labels(ax, labels)
        d = next((int(r["collapse"][r["collapse"].representation == rep].d.iloc[0]) for r in runs.values()
                  if (r["collapse"].representation == rep).any()), None)
        ax.set_title(REP_TITLE[rep] + (f", d = {d}" if d else ""), fontsize=7.5, color=INK, loc="left")
        ax.set_xlabel("Epoch", fontsize=7)
    axes[0].set_ylabel(ylabel, fontsize=7)
    axes[3].set_ylabel(ylabel, fontsize=7)
    stage_legend(fig, list(runs))
    if note:
        fig.text(0.0, -0.02, note, fontsize=6.5, color=MUTED, va="top")
    fig.tight_layout(h_pad=1.5, w_pad=2.5)
    show(fig)


def trajectory_figure(runs):
    """F5b: alignment against mean cosine, one point per epoch (start marked o, end marked ■)."""
    fig, axes = grid(len(REPS))
    for ax, rep in zip(axes, REPS):
        labels = []
        for st, r in runs.items():
            c = r["collapse"][r["collapse"].representation == rep]
            if c.empty:
                continue
            g = c.groupby("epoch")[["mean_cosine", "alignment"]].mean()  # mean over seeds
            colour, ls, _ = STAGES[st]
            ax.plot(g.mean_cosine, g.alignment, color=colour, linestyle=ls, linewidth=1.1)
            ax.plot(g.mean_cosine.iloc[0], g.alignment.iloc[0], "o", color=colour, markersize=4,
                    markerfacecolor="white")
            ax.plot(g.mean_cosine.iloc[-1], g.alignment.iloc[-1], "s", color=colour, markersize=4)
            labels.append((g.mean_cosine.iloc[-1], g.alignment.iloc[-1], st))
        if not labels:
            ax.axis("off")
            ax.text(0.5, 0.5, "not measured\nin these stages", ha="center", va="center", fontsize=7, color=MUTED,
                    transform=ax.transAxes)
        else:
            style(ax)
            point_labels(ax, labels, dx=5)  # next to the last-epoch marker
        ax.set_title(REP_TITLE[rep], fontsize=7.5, color=INK, loc="left")
        ax.set_xlabel("Mean cosine between windows", fontsize=7)
    axes[0].set_ylabel("Alignment with augmented view", fontsize=7)
    axes[3].set_ylabel("Alignment with augmented view", fontsize=7)
    stage_legend(fig, list(runs))
    fig.text(0.0, -0.02, "o = epoch 0 (untrained), ■ = last epoch. Good: low cosine (windows spread), "
             "high alignment (augmented views stay close).", fontsize=6.5, color=MUTED, va="top")
    fig.tight_layout(h_pad=1.5, w_pad=2.5)
    show(fig)


def spectrum_figure(st, r, epochs):
    """F3: per stage, the normalised eigenvalue spectrum and cumulative variance at chosen epochs."""
    reps = [rep for rep in REPS if (r["spectra"].representation == rep).any()]
    present = set(r["spectra"].epoch)
    missing = [e for e in epochs if e not in present]
    if missing:
        raise ValueError(f"stage {st}: spectrum_epochs {missing} not in {r['folder']}")
    # Epochs as a sequential ramp of one hue, light (early) to dark (late).
    ramp = plt.get_cmap("Blues")(np.linspace(0.35, 1.0, len(epochs)))
    fig, axes = plt.subplots(len(reps), 2, figsize=(6.3, 0.4 + 2.0 * len(reps)), squeeze=False)
    seed = r["seeds"][0]  # spectra of one seed: a mean of spectra over seeds would hide their shape
    sp = r["spectra"][r["spectra"].seed == seed]
    for row, rep in zip(axes, reps):
        for e, colour in zip(epochs, ramp):
            ev = np.sort(sp[(sp.representation == rep) & (sp.epoch == e)].eigenvalue.to_numpy())[::-1]
            k = np.arange(1, len(ev) + 1)
            row[0].plot(k, ev / ev.sum(), color=colour, linewidth=1.1, label=f"epoch {e}")
            row[1].plot(k, np.cumsum(ev) / ev.sum(), color=colour, linewidth=1.1)
        row[0].set_xscale("log"), row[0].set_yscale("log")
        row[0].set_ylim(bottom=1e-8)  # eigenvalues below this are numerical zeros
        row[1].set_xscale("log")
        row[1].axhline(0.9, color=MUTED, linestyle=":", linewidth=1)
        row[0].set_title(f"{REP_TITLE[rep]}: eigenvalue / total", fontsize=7.5, color=INK, loc="left")
        row[1].set_title(f"{REP_TITLE[rep]}: cumulative share of variance", fontsize=7.5, color=INK, loc="left")
        for ax in row:
            style(ax)
    for ax in axes[-1]:
        ax.set_xlabel("Direction k (largest eigenvalue first)", fontsize=7)
    fig.tight_layout(h_pad=1.2, w_pad=2.0)
    handles, texts = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, texts, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=len(epochs), fontsize=6.5,
               frameon=False)
    fig.suptitle(f"Stage {STAGE_TEXT[st]}; seed {seed}", fontsize=8, color=INK, x=0.0, ha="left", y=1.02)
    show(fig)


def final_spectrum_figure(runs):
    """F3b: cumulative share of variance at the last epoch, stages overlaid, one panel per representation."""
    fig, axes = grid(len(REPS))
    for ax, rep in zip(axes, REPS):
        labels = []
        for st, r in runs.items():
            sp = r["spectra"][(r["spectra"].representation == rep) & (r["spectra"].seed == r["seeds"][0])]
            if sp.empty:
                continue
            ev = np.sort(sp[sp.epoch == sp.epoch.max()].eigenvalue.to_numpy())[::-1]
            cum = np.cumsum(ev) / ev.sum()
            k = np.arange(1, len(ev) + 1)
            ax.plot(k, cum, color=STAGES[st][0], linestyle=STAGES[st][1], linewidth=1.3)
            labels.append((k[9], cum[9], st))  # label at k = 10, where the curves differ most visibly
        if not labels:
            ax.axis("off")
            ax.text(0.5, 0.5, "not measured\nin these stages", ha="center", va="center", fontsize=7, color=MUTED,
                    transform=ax.transAxes)
        else:
            ax.set_xscale("log")
            ax.axhline(0.9, color=MUTED, linestyle=":", linewidth=1)
            style(ax)
            point_labels(ax, labels, dx=-4, ha="right")
        ax.set_title(REP_TITLE[rep], fontsize=7.5, color=INK, loc="left")
        ax.set_xlabel("Number of directions k", fontsize=7)
    axes[0].set_ylabel("Share of variance in top k", fontsize=7)
    axes[3].set_ylabel("Share of variance in top k", fontsize=7)
    stage_legend(fig, list(runs))
    fig.text(0.0, -0.02, "Last epoch, first seed. A curve that rises steeply = variance in few directions "
             "(dimensional collapse). Dotted line: 90 %.", fontsize=6.5, color=MUTED, va="top")
    fig.tight_layout(h_pad=1.5, w_pad=2.5)
    show(fig)


def probe_figure(runs, references):
    """F7: test accuracy against label budget, per probe and representation; other methods as reference lines."""
    probes = [p for p in ("logreg", "svm_ts2vec")]  # D26 probes, same order in every figure
    probe_title = {"logreg": "logistic regression", "svm_ts2vec": "SVM (TS2Vec protocol)"}
    reps = [rep for rep in REPS if any(r["probes"] is not None and (r["probes"].representation == rep).any()
                                       for r in runs.values())]
    if not reps:
        print("  no probes.csv in any run: probe figure skipped", flush=True)
        return
    ref = {name: pd.read_csv(os.path.join(f, "results.csv")) for name, f in references.items()}
    fig, axes = plt.subplots(len(probes), len(reps), figsize=(6.3, 2.0 * len(probes)), squeeze=False, sharey=True)
    pending = []  # direct labels, placed once the shared y axis is final
    for i, p in enumerate(probes):
        for j, rep in enumerate(reps):
            ax = axes[i][j]
            for (name, res), ls in zip(ref.items(), ("--", ":")):  # same reference lines in every panel (named below)
                g = res[res.probe == p].groupby("label_fraction").accuracy.mean()
                ax.plot(np.log10(g.index * 100), g, color=MUTED, linestyle=ls, linewidth=1)
            labels = []
            for st, r in runs.items():
                if r["probes"] is None:
                    continue
                q = r["probes"][(r["probes"].representation == rep) & (r["probes"].probe == p)]
                if q.empty:
                    continue
                m, lo, hi = over_seeds(q, "label_fraction", "accuracy")
                x = np.log10(m.index * 100)
                draw_line(ax, x, m, lo, hi, st, q.seed.nunique(), marker="o", markersize=3)
                labels.append((x[0], m.iloc[0], st))
            ax.set_xticks([0, 1, 2], ["1 %", "10 %", "100 %"])
            ax.set_xlim(-0.35, 2.45)
            pending.append((ax, labels))
            ax.set_title(f"{REP_TITLE[rep].split(' (')[0]}, {probe_title[p]}", fontsize=7, color=INK, loc="left")
            style(ax)
        axes[i][0].set_ylabel("Test accuracy", fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("Training labels used by the probe", fontsize=7)
    for ax, labels in pending:
        point_labels(ax, labels, dx=-5, ha="right")  # left of the 1 % point
    stage_legend(fig, [st for st, r in runs.items() if r["probes"] is not None])
    fig.text(0.0, -0.02, "Grey lines: " + ", ".join(f"{n} ({s})" for n, s in zip(references, ("dashed", "dotted")))
             + ", mean over their seeds, same probe.", fontsize=6.5, color=MUTED, va="top")
    fig.tight_layout(h_pad=1.5, w_pad=1.0)
    show(fig)


def pca_figure(st, r, device):
    """2-D PCA of the diagnostic windows at epoch 0 (rebuilt from the seed) and the last epoch, coloured by class.

    Labels are used only for colouring. The rebuilt models are checked against the stored
    spectra (largest eigenvalue) before anything is drawn.
    """
    import torch

    from harness import runs as harness_runs, transfer
    from harness.data import canonical
    from harness.methods import tfc
    from studies.tfc_ablation.ablation import build_model
    from studies.tfc_ablation.collapse import diagnostic_rows, represent

    cfg = r["cfg"]
    seed = r["seeds"][0]
    meta = canonical.load_metadata(cfg["data_root"], cfg["dataset"])
    X, labels, _ = transfer.load_set(cfg, meta, cfg["run_id"], cfg["train_split"])
    rows = diagnostic_rows(len(X), cfg["diagnostics"]["windows"], cfg["diagnostics"]["seed"])
    x = tfc.to_device(X[rows], device)
    y = np.searchsorted(np.unique(labels), labels[rows])
    reps = cfg["diagnostics"]["representations"]

    harness_runs.set_seed(seed)  # same random draws as ablation.py before the model is built
    model0 = build_model(st, x.shape[2], x.shape[1], cfg["tfc"]["encoder"]).to(device)
    last = build_model(st, x.shape[2], x.shape[1], cfg["tfc"]["encoder"])
    last.load_state_dict(torch.load(os.path.join(r["folder"], f"seed_{seed}", "pretrained.pt"), map_location="cpu"))
    last = last.to(device)
    sp = r["spectra"][r["spectra"].seed == seed]
    epochs = (0, int(sp.epoch.max()))

    side = min(1.9, 6.3 / len(reps))  # square panels, at most the page width in total
    fig, axes = plt.subplots(2, len(reps), figsize=(side * len(reps) + 0.4, 2 * side + 0.5), squeeze=False)
    order = np.random.default_rng(0).permutation(len(y))  # draw the classes interleaved, not one on top
    for i, (epoch, model) in enumerate(zip(epochs, (model0, last))):
        worst = 0.0  # largest relative difference to the stored largest eigenvalue
        R = represent(model, x, tfc.spectrum(x), cfg["tfc"]["eval_chunk"])
        for j, rep in enumerate(reps):
            Z = R[rep].double().cpu()
            Zc = Z - Z.mean(0)
            _, S, Vh = torch.linalg.svd(Zc, full_matrices=False)
            ev = S.square() / (len(Z) - 1)  # covariance eigenvalues, as diagnostics.covariance_eigenvalues
            stored = sp[(sp.representation == rep) & (sp.epoch == epoch)].eigenvalue.max()
            if not np.isclose(float(ev[0]), stored, rtol=1e-2):
                raise RuntimeError(f"stage {st}, {rep}, epoch {epoch}: rebuilt model gives largest eigenvalue "
                                   f"{float(ev[0]):.6g}, the run stored {stored:.6g}; the picture would not be "
                                   "the run's model")
            worst = max(worst, abs(float(ev[0]) / stored - 1))
            P = (Zc @ Vh[:2].T).numpy()
            share = (ev[:2] / ev.sum()).numpy()
            ax = axes[i][j]
            ax.scatter(P[order, 0], P[order, 1], c=np.array(CLASS_COLOURS)[y[order]], s=1.5, alpha=0.5,
                       linewidths=0, rasterized=True)
            ax.set_title(f"{REP_TITLE[rep].split(' (')[0]}, epoch {epoch}", fontsize=7, color=INK, loc="left")
            ax.set_xlabel(f"PC1 ({share[0]:.0%})", fontsize=6.5)
            ax.set_ylabel(f"PC2 ({share[1]:.0%})", fontsize=6.5)
            style(ax)
            ax.set_box_aspect(1)
            ax.tick_params(labelsize=6)
        print(f"  stage {st}, epoch {epoch}: rebuilt model matches the stored spectra "
              f"(largest eigenvalue within {worst:.1e} relative)", flush=True)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=c, markersize=4, label=f"class {k}")
               for k, c in zip(np.unique(labels), CLASS_COLOURS)]
    fig.tight_layout(h_pad=1.2, w_pad=1.5)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=6.5, frameon=False)
    fig.suptitle(f"Stage {STAGE_TEXT[st]}; seed {seed}\nfirst two principal components of the {len(y)} diagnostic windows",
                 fontsize=7.5, color=INK, x=0.0, ha="left", y=1.08)
    show(fig)


def load(config_path):
    """Read the figure config and all runs listed in it: (config, {stage: run})."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    for key in ("runs", "references", "variance_share", "spectrum_epochs", "pca_device"):
        if cfg.get(key) is None:
            raise ValueError(f"{key} is not set (placeholder)")
    unknown = set(cfg["runs"]) - set(STAGES)
    if unknown:
        raise ValueError(f"runs: unknown stages {sorted(unknown)}")
    runs = {st: load_run(cfg["runs"][st], cfg["variance_share"]) for st in STAGES if st in cfg["runs"]}
    return cfg, runs
