"""Literature protocols that a config names instead of writing out its split (D216).

paderborn_lessmeier(k): Lessmeier et al. (2016), Section 7.3, combination k of 10 (D220, D222, D225).
A config says  split: {protocol: paderborn_lessmeier}  and scripts/pretrain.py --combination k
replaces that with the train / test / pretrain_only filters returned here.
"""

from itertools import combinations

# Lessmeier et al. (2016), Table 10: five bearings per class, in the table's row order (D220).
LESSMEIER_7_3 = {
    "healthy": ["K001", "K002", "K003", "K004", "K005"],
    "OR": ["KA04", "KA15", "KA16", "KA22", "KA30"],
    "IR": ["KI04", "KI14", "KI16", "KI18", "KI21"],
}
LESSMEIER_7_3_BEARINGS = [b for column in LESSMEIER_7_3.values() for b in column]
SETTING_0 = "N15_M07_F10"  # 1500 rpm, 0.7 Nm, 1000 N: probes and test (D222)
# The 10 combinations: which three rows of Table 10 are training, the same rows for all three classes
# (inferred, D220: choosing per class independently would give 1,000 combinations, not 10).
COMBINATIONS = list(combinations(range(5), 3))  # k = 0..9, in itertools order: (0,1,2), (0,1,3), ...


def paderborn_lessmeier(k):
    """The split of combination k: filters on the captures table for each part (D147).

    train: the three training rows of each class at setting 0; test: the other two rows at setting 0.
    pretrain_only: none; pretraining uses the training windows only (D225). Settings 1-3 and the
    bearings outside Table 10 are not used.
    """
    train_rows = COMBINATIONS[k]
    train = [column[i] for column in LESSMEIER_7_3.values() for i in train_rows]
    test = [b for b in LESSMEIER_7_3_BEARINGS if b not in train]
    return {
        "train": {"bearing": train, "condition": SETTING_0},
        "test": {"bearing": test, "condition": SETTING_0},
        "pretrain_only": {"bearing": []},  # selects no captures (D225)
    }


PROTOCOLS = {"paderborn_lessmeier": paderborn_lessmeier}
