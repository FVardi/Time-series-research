"""Write the pilot's run configs (D182-D185, D197, D198) to studies/split_pilot/configs/.

Run once: uv run python studies/split_pilot/make_configs.py
One config per variant and split. Variants: 0 = plain TS2Vec at batch 64 (D183 item 6, D185 item 2);
A = split + op heads; B = A + adversary; C = A + HSIC; B and C at removal weights 0.1, 1, 10 (D185);
B1live, C100, C1000 on departures and random only (D207).
The dataset sections are written out below in full, so each config can be read on its own.
Existing configs are left as they are if unchanged; a changed one stops the script, unless it is
run with --update, which rewrites the changed configs and lists them.
"""

import os

OUT = os.path.join(os.path.dirname(__file__), "configs")

# ---- dataset and split sections (everything above 'method:') ----
SYNTHETIC = """dataset: {dataset}
data_root: data/canonical
channels: [AE, US]
labels: {labels}
windows:
  length: 1024
  sample_rate_hz: 10000  # the generated rate (D196)
  standardise: true  # per channel, statistics of the pretraining windows (as D137)
conditions: [speed_rpm, temperature_c]  # one head per factor (D197, D198)
split:
{split}
"""
SYN_SPLITS = {
    # random windows, the default (D198 item 1); random_state as the CWRU split (D174)
    "synthetic_random": ("synthetic_crossed", "synthetic",
                         "  captures: {}\n  random_windows: {test_fraction: 0.2, random_state: 40}"),
    # departures (D198 item 1): probe on nominal runs, test on the departure runs of bearings B0, B1;
    # the departure runs of B2, B3 are pretraining-only (unlabelled)
    "synthetic_departures": ("synthetic_departures", "synthetic_shared",
                             "  train: {departure: [false]}\n"
                             "  test: {departure: [true], bearing: [B0, B1]}\n"
                             "  pretrain_only: {departure: [true], bearing: [B2, B3]}"),
    # the same test, pretraining without any departure runs (D198 item 2): purely confounded pretraining data
    "synthetic_departures_nominalpre": ("synthetic_departures", "synthetic_shared",
                                        "  train: {departure: [false]}\n"
                                        "  test: {departure: [true], bearing: [B0, B1]}\n"
                                        "  pretrain_only: {departure: [true], bearing: []}  # none"),
}
# leave one speed out (D198 item 1b, D199: all four speeds in turn): test = every run at that speed
SPEEDS = [600, 1200, 1800, 2400]
for held_out in SPEEDS:
    others = ", ".join(str(v) for v in SPEEDS if v != held_out)
    SYN_SPLITS[f"synthetic_loso{held_out}"] = (
        "synthetic_crossed", "synthetic_shared",
        f"  train: {{speed_rpm: [{others}]}}\n"
        f"  test: {{speed_rpm: [{held_out}]}}\n"
        f"  pretrain_only: {{speed_rpm: []}}  # none")
SYN_EVAL = """evaluation:
  label_fractions: [0.01, 0.1, 1.0]
  probes: [logreg, floor]
  cv_groups: false  # plain stratified 5-fold (as D175)
  published_accuracy: null
"""

CWRU_COMMON = """dataset: cwru
data_root: data/canonical
channels: [DE, FE]  # D131
windows:  # D136, D137
  length: 2048
  sample_rate_hz: 12000  # normal files (48 kHz) resampled to 12 kHz
  standardise: true  # per channel, statistics of the pretraining windows (D137)
conditions: [load_hp]  # D182, D198
"""
CWRU_SPLITS = {
    "cwru": ("labels: cwru  # D118, D125\n"
             "split:  # random windows (D163, D164, D171, D174)\n"
             "  captures: {sample_rate_hz: 12000, fault_end: [DE, FE], fault_size_mil: [7, 14, 21]}\n"
             "  random_windows: {test_fraction: 0.2, random_state: 40}\n"
             "  pretrain_only: {fault_end: [none]}\n",
             "evaluation:  # as configs/ts2vec_cwru.yaml\n"
             "  label_fractions: [0.01, 0.1, 1.0]\n  probes: [logreg, floor]\n"
             "  cv_groups: false  # D175\n  published_accuracy: null\n"),
    "cwru_bearing": ("labels: cwru  # D118, D125\n"
                     "split:  # bearing split (D128, D140, D147)\n"
                     "  train: {sample_rate_hz: 12000, fault_end: [DE, FE], fault_size_mil: [7, 14]}\n"
                     "  test: {sample_rate_hz: 12000, fault_end: [DE, FE], fault_size_mil: [21]}\n"
                     "  pretrain_only: {fault_end: [none]}  # D126\n",
                     "evaluation:  # as configs/ts2vec_cwru_bearing.yaml\n"
                     "  label_fractions: [0.01, 0.1, 1.0]\n  probes: [logreg, floor]\n"
                     "  cv_groups: true  # D129, D148\n  published_accuracy: null\n"),
}

# ---- method sections ----
TS2VEC = """  repr_dims: 320
  hidden_dims: 64
  depth: 10
  lr: 0.001
  batch_size: 64  # D183 item 6 (official: 8)
  max_train_length: 3000
  temporal_unit: 0
  n_iters: 600  # D185 item 2
"""
def seeds_of(name, variant, synthetic):
    """Every config keeps the full seeds; exploration runs pick seed 0 with --seeds 0 (D201, D204)."""
    return "[0, 1, 2]"


VARIANTS = {"0": None, "A": ("none", None), "B0.1": ("adversary", 0.1), "B1": ("adversary", 1.0),
            "B10": ("adversary", 10.0), "C0.1": ("hsic", 0.1), "C1": ("hsic", 1.0), "C10": ("hsic", 10.0),
            # D207: diagnostics after the first block probes; only on the splits in ONLY_SPLITS
            "B1live": ("adversary", 1.0), "C100": ("hsic", 100.0), "C1000": ("hsic", 1000.0)}
LIVE = {"B1live"}  # also store the live encoder and its representations (D207)
ONLY_SPLITS = {v: {"synthetic_departures", "synthetic_random"} for v in ("B1live", "C100", "C1000")}


def method_section(variant):
    if VARIANTS[variant] is None:  # variant 0: the wrapped official TS2Vec
        return "method: ts2vec\n", "ts2vec:  # official settings (D28) except batch size (D183, D185)\n" + TS2VEC
    removal, weight = VARIANTS[variant]
    extra = (f"  z_op_dims: 32  # D183 item 1\n  head_hidden: 64  # D185 item 5\n  op_weight: 1.0  # D185 item 4\n"
             f"  removal: {removal}  # D182 item 3\n")
    if removal != "none":
        extra += f"  removal_weight: {weight}  # {'D207' if variant in ONLY_SPLITS else 'D185 item 4'}\n"
    if removal == "adversary":
        extra += "  ramp_max: 1.0  # D185 item 3\n"
    if variant in LIVE:
        extra += "  save_live_encoder: true  # D207: also the live encoder's representations (diagnostic)\n"
    return "method: ts2vec_split\n", "ts2vec_split:  # D183, D185, D197\n" + TS2VEC + extra


def write(name, body, update=False):
    path = os.path.join(OUT, f"{name}.yaml")
    if os.path.exists(path):
        if open(path, encoding="utf-8").read() == body:
            return False
        if not update:
            raise FileExistsError(f"{path} exists with other content")
        print(f"updated {name}")
    with open(path, "w", encoding="utf-8") as f:
        f.write(body)
    return True


def main(update=False):
    os.makedirs(OUT, exist_ok=True)
    n = 0
    for variant in VARIANTS:
        method, settings = method_section(variant)
        common = "device: cuda\nrankme_epsilon: 1.0e-7  # D152, diagnostic only\n\n"
        for split, (dataset, labels, split_text) in SYN_SPLITS.items():
            if variant in ONLY_SPLITS and split not in ONLY_SPLITS[variant]:
                continue
            name = f"pilot_{variant}_{split}"
            seeds = f"seeds: {seeds_of(name, variant, synthetic=True)}\n"
            data = SYNTHETIC.format(dataset=dataset, labels=labels, split=split_text)
            if variant == "0":  # plain TS2Vec takes no conditions (D198)
                data = data.replace("conditions: [speed_rpm, temperature_c]  # one head per factor (D197, D198)\n", "")
            n += write(name, f"# Split-representation pilot (D182), variant {variant}, {split}. Written by make_configs.py.\n"
                             f"name: {name}\n" + data + method + seeds + common + settings + "\n" + SYN_EVAL, update)
        for split, (split_text, eval_text) in CWRU_SPLITS.items():
            if variant in ONLY_SPLITS and split not in ONLY_SPLITS[variant]:
                continue
            name = f"pilot_{variant}_{split}"
            seeds = f"seeds: {seeds_of(name, variant, synthetic=False)}\n"
            data = CWRU_COMMON if variant != "0" else CWRU_COMMON.replace("conditions: [load_hp]  # D182, D198\n", "")
            n += write(name, f"# Split-representation pilot (D182), variant {variant}, {split}. Written by make_configs.py.\n"
                             f"name: {name}\n" + data + split_text + method + seeds + common + settings + "\n" + eval_text,
                       update)
    print(f"Done: {n} new configs in {OUT}")


if __name__ == "__main__":
    import sys
    main(update="--update" in sys.argv[1:])
