"""Class labels and groups, derived per dataset from the captures table (D147).

Label derivation lives here, not in the adapters (D4). Each function takes the captures
table and returns two lists, one entry per capture:
  label: the class, or None for captures without a class (e.g. unlabelled pretraining data)
  group: the unit that must stay on one side of the split, or None (D121)
"""


def cwru(captures):
    """CWRU (D118, D125): class = fault location, outer race with its clock hour ('outer@6');
    group = the faulty bearing's ID. Normal files (no fault) have neither: their healthy
    bearings are background, not test bearings (D125)."""
    labels, groups = [], []
    for c in captures.itertuples():
        if c.fault_location == "none":
            labels.append(None)
            groups.append(None)
            continue
        labels.append(f"outer@{int(c.clock_hour)}" if c.fault_location == "outer" else c.fault_location)
        groups.append(c.de_bearing if c.fault_end == "DE" else c.fe_bearing)
    return labels, groups


LABELS = {"cwru": cwru}
