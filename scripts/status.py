"""Write STATUS.md: what has run, from the configs and run folders (D261).

Usage (from the repository root; needs only PyYAML):
    uv run python scripts/status.py

STATUS.md is generated in full, so it is never out of date and is never edited by hand.
Plans and to-dos live in the project checklist; current rules in docs/; decisions in decisions/.

How a run is matched to a config: a run folder is runs/<YYYYmmdd_HHMMSS>_<config name>
(Paderborn: one folder per combination, <name>_c<k>, plus <name>_combinations.csv when all
combinations are done, D220). The benchmark columns are the dataset parts in configs/datasets/
(D260); a config <method>_<column>.yaml is that column's standard config (D239), any longer name
(<method>_<column>_<suffix>) a variant or diagnostic. configs/compare/<column>.yaml, where it
exists, names the standard run by hand (D203) and overrides the automatic choice.
"""

import glob
import os
import re
from datetime import datetime

import yaml

from harness import runs

# Row order of the progress table: SSL methods, then baselines (D75, D77). Methods not listed
# here are added at the end in alphabetical order.
METHOD_ORDER = ["ts2vec", "tloss", "tstcc", "tsjepa", "patchtst", "tfc", "tfjepa", "simmtm", "frera",
                "raw", "fft", "envelope", "catch22", "minirocket", "mssp", "pooled"]
RUN_NAME = re.compile(r"^(\d{8})_(\d{6})_(.+?)(_c(\d+))?$")  # timestamp, config name, Paderborn combination


def split_name(name):
    """(row, column, variant) of a config name <row>_<column>[_<variant>], or None.

    The columns are the dataset parts in configs/datasets/; the row is usually the method
    (mssp_herning runs the 'raw' method on stored MSSP statistics, so it gets its own row).
    """
    columns = sorted((os.path.basename(p)[:-5] for p in glob.glob("configs/datasets/*.yaml")),
                     key=len, reverse=True)  # longest first, so xjtu_sy_stage wins over xjtu_sy
    for column in columns:
        if name.endswith("_" + column):
            return name[:-len(column) - 1], column, ""
        if "_" + column + "_" in name:
            row, variant = name.split("_" + column + "_", 1)
            return row, column, variant
    return None


def configs():
    """{config name: {"method" (= row), "column", "variant"}} for every configs/*.yaml that runs a method."""
    out = {}
    for path in sorted(glob.glob("configs/*.yaml")):
        name = os.path.basename(path)[:-5]
        parts = split_name(name)
        if parts is None or "method" not in runs.load_config(path):
            continue  # not a method run (e.g. a check whose config lives elsewhere)
        out[name] = {"method": parts[0], "column": parts[1], "variant": parts[2]}
    return out


def run_folders():
    """{config name: [runs, newest first]}; each run {"folder", "date", "done", "combinations"}."""
    found = {}
    for entry in sorted(os.listdir("runs"), reverse=True):  # newest first
        m = RUN_NAME.match(entry)
        if not m or not os.path.isdir(os.path.join("runs", entry)):
            continue
        name, combination = m.group(3), m.group(5)
        date = f"{m.group(1)[:4]}-{m.group(1)[4:6]}-{m.group(1)[6:]}"
        done = os.path.exists(os.path.join("runs", entry, "results.csv"))
        if combination is None:
            found.setdefault(name, []).append({"folder": f"runs/{entry}", "date": date, "done": done})
        else:  # Paderborn combination: count it towards the config's combined run
            found.setdefault(name + "#c", []).append({"folder": f"runs/{entry}", "date": date, "done": done})
    for path in sorted(glob.glob("runs/*_combinations.csv"), reverse=True):
        m = re.match(r"^(\d{8})_\d{6}_(.+)_combinations\.csv$", os.path.basename(path))
        date = f"{m.group(1)[:4]}-{m.group(1)[4:6]}-{m.group(1)[6:]}"
        found.setdefault(m.group(2), []).insert(0, {"folder": path.replace(os.sep, "/"), "date": date, "done": True})
    return found


def state(name, found):
    """(symbol, date, folder) of a config's newest finished run, or of what exists instead."""
    finished = [r for r in found.get(name, []) if r["done"]]
    newest = (found.get(name) or [None])[0]
    if finished and newest is not finished[0]:  # a newer run exists that has no results (yet)
        return f"✓ {finished[0]['date'][5:]}, newer ◐ {newest['date'][5:]}", "", finished[0]["folder"]
    if finished:
        return "✓", finished[0]["date"], finished[0]["folder"]
    parts = found.get(name + "#c", [])
    if parts:  # Paderborn combinations started, table not written yet
        n = len({re.search(r"_c(\d+)$", r["folder"]).group(1) for r in parts if r["done"]})
        return f"◐ {n}/10", parts[0]["date"], parts[0]["folder"]
    started = found.get(name, [])
    if started:  # folder without results: encoded but not evaluated, or running, stopped or failed
        encoded = glob.glob(os.path.join(started[0]["folder"], "seed_*", "Z_test.npy"))
        return ("◐ not evaluated" if encoded else "◐"), started[0]["date"], started[0]["folder"]
    return "○", "", ""


def compare_overrides():
    """{(method, column): run folder} from configs/compare/<column>.yaml (D203)."""
    out = {}
    for path in glob.glob("configs/compare/*.yaml"):
        column = os.path.basename(path)[:-5]
        for entries in runs.load_config(path)["settings"].values():
            for entry in entries.values():
                cfg = runs.load_config(os.path.join(entry["run"], "config.yaml"))
                parts = split_name(cfg["name"])
                out[(parts[0] if parts else cfg["method"], column)] = entry["run"]
    return out


def cells(cfgs, found):
    """{(method, column): (text, folder)} for the standard configs (D239); compare configs win (D203)."""
    overrides = compare_overrides()
    out = {}
    for name, c in cfgs.items():
        if c["variant"]:
            continue
        key = (c["method"], c["column"])
        if key in overrides:
            out[key] = ("✓ (compare)", overrides[key])
        else:
            symbol, date, folder = state(name, found)
            out[key] = (f"{symbol} {date[5:]}".strip(), folder)
    for key, folder in overrides.items():  # a compared run whose config was renamed since
        out.setdefault(key, ("✓ (compare)", folder))
    return out


def progress_table(cfgs, table):
    """Markdown table: methods x benchmark columns."""
    columns = sorted({column for _, column in table})
    methods = sorted({method for method, _ in table},
                     key=lambda m: (METHOD_ORDER.index(m) if m in METHOD_ORDER else 99, m))
    lines = ["| method | " + " | ".join(columns) + " |", "|---" * (len(columns) + 1) + "|"]
    for method in methods:
        row = [table.get((method, column), ("–", ""))[0] for column in columns]
        lines.append(f"| {method} | " + " | ".join(row) + " |")
    used = sorted((column, method, folder) for (method, column), (_, folder) in table.items() if folder)
    return lines, used


def variants_list(cfgs, found):
    """Per column, the variant and diagnostic configs (longer names) with their state."""
    lines = []
    for column in sorted({c["column"] for c in cfgs.values()}):
        items = [(n, c) for n, c in cfgs.items() if c["column"] == column and c["variant"]]
        if not items:
            continue
        cells = []
        for name, _ in sorted(items):
            symbol, date, _ = state(name, found)
            cells.append(f"`{name}` {symbol} {date[5:]}".strip())
        lines.append(f"- **{column}**: " + "; ".join(cells))
    return lines


def queue_logs():
    """Last job and failures of each runs/queue_*.log (written by PowerShell, so maybe UTF-16)."""
    lines = []
    for path in sorted(glob.glob("runs/queue_*.log")):
        raw = open(path, "rb").read()
        text = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8", "replace")
        jobs = [l.strip() for l in text.splitlines() if l.startswith("===")]
        failed = [l.strip() for l in text.splitlines() if l.startswith("FAILED")]
        changed = datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M")
        lines.append(f"- `{path}` (last written {changed}): last entry `{jobs[-1] if jobs else '-'}`"
                     + (f"; failed: {', '.join(f'`{f[7:]}`' for f in failed)}" if failed else ""))
    return lines


def datasets_table(table):
    """Converted datasets, their card in docs/ and the number of standard runs done."""
    names = sorted(set(os.listdir("data/canonical")) if os.path.isdir("data/canonical") else set())
    cards = {os.path.basename(p)[:-3] for p in glob.glob("docs/datasets/*.md")}
    lines = ["| dataset | converted | card | standard configs | standard runs done |", "|---|---|---|---|---|"]
    for ds in sorted(set(names) | cards):
        std = [text for (_, column), (text, _) in table.items() if column == ds or column.startswith(ds + "_")]
        done = sum(text.startswith("✓") for text in std)
        card = f"[card](docs/datasets/{ds}.md)" if ds in cards else "–"
        lines.append(f"| {ds} | {'yes' if ds in names else 'no'} | {card} | {len(std)} | {done} |")
    return lines


def studies_list():
    """studies/<name>/README.md: title and the 'Status:' line, if any."""
    lines = []
    for readme in sorted(glob.glob("studies/*/README.md")):
        text = open(readme, encoding="utf-8").read().splitlines()
        title = text[0].lstrip("# ").strip() if text else ""
        status = next((l.split(":", 1)[1].strip() for l in text if l.startswith("Status:")), "no status line")
        lines.append(f"- [{title}]({readme.replace(os.sep, '/')}): {status}")
    return lines


def main():
    cfgs, found = configs(), run_folders()
    table = cells(cfgs, found)
    progress, used = progress_table(cfgs, table)
    archived = len(os.listdir("runs/archive")) if os.path.isdir("runs/archive") else 0
    out = [
        "# STATUS — what has run",
        "",
        f"Generated by `scripts/status.py` on {datetime.now():%Y-%m-%d %H:%M} from `configs/` and `runs/` (D261).",
        "Do not edit by hand: run `uv run python scripts/status.py` again. To-dos: the project checklist;",
        "current rules: `docs/`; decisions: `DECISIONS.md`; supervisor questions: `QUESTIONS.md`.",
        "",
        "## Benchmark progress (standard runs, D239: `configs/<method>_<dataset>.yaml`, seed 0)",
        "",
        "✓ finished (date of the newest finished run) · ◐ not evaluated: encoded, `evaluate.py` not run yet ·",
        "◐ started, nothing stored yet (running, stopped or failed; Paderborn: combinations finished of 10) ·",
        "○ config ready, not run · – no config.",
        "✓ (compare): the run named in `configs/compare/<dataset>.yaml` (D203).",
        "",
        *progress,
        "",
        "<details><summary>Run folder behind each cell</summary>",
        "",
        *[f"- {column} / {method}: `{folder}`" for column, method, folder in used],
        "",
        "</details>",
        "",
        "## Variants and diagnostics (outside the benchmark tables, D239)",
        "",
        *variants_list(cfgs, found),
        "",
        "## Queues",
        "",
        *(queue_logs() or ["- none"]),
        "",
        "## Datasets",
        "",
        *datasets_table(table),
        "",
        "## Studies",
        "",
        *(studies_list() or ["- none"]),
        "",
        f"Archived run folders (`runs/archive/`): {archived}.",
        "",
    ]
    with open("STATUS.md", "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(out))
    print("Wrote STATUS.md")


if __name__ == "__main__":
    main()
