"""Run folders (D99) and configs composed from shared parts (D260)."""

import glob
import os

import yaml

from harness import runs


def make_run(tmp_path, monkeypatch, uncommitted):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runs, "git_commit", lambda: {"commit": "abc1234", "uncommitted_changes": uncommitted})
    (tmp_path / "config.yaml").write_text("name: test\n")
    return runs.new_run_dir("config.yaml", {"name": "test"})


def test_warns_on_uncommitted_changes(tmp_path, monkeypatch, capsys):
    make_run(tmp_path, monkeypatch, uncommitted=True)
    assert "uncommitted changes" in capsys.readouterr().err


def test_silent_on_clean_tree(tmp_path, monkeypatch, capsys):
    make_run(tmp_path, monkeypatch, uncommitted=False)
    assert capsys.readouterr().err == ""


def test_parts_merge_in_order(tmp_path):
    """defaults: parts merged in order, the file's own keys on top; mappings merged, lists replaced."""
    (tmp_path / "datasets").mkdir()
    (tmp_path / "methods").mkdir()
    (tmp_path / "datasets" / "d.yaml").write_text("dataset: d\nevaluation: {probes: [logreg], label_fractions: [0.1, 1.0]}\n")
    (tmp_path / "methods" / "m.yaml").write_text("method: m\nm: {a: 1, b: {c: 2}}\n")
    (tmp_path / "m_d.yaml").write_text("defaults: [datasets/d.yaml, methods/m.yaml]\nname: m_d\n"
                                       "m: {b: {d: 3}}\nevaluation: {probes: [svm]}\n")
    cfg = runs.load_config(str(tmp_path / "m_d.yaml"))
    assert cfg == {"dataset": "d", "evaluation": {"probes": ["svm"], "label_fractions": [0.1, 1.0]},
                   "method": "m", "m": {"a": 1, "b": {"c": 2, "d": 3}}, "name": "m_d"}


def test_run_folder_holds_resolved_config_and_sources(tmp_path, monkeypatch):
    """config.yaml in the run folder = the resolved config; the files it came from are in config_sources/."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runs, "git_commit", lambda: {"commit": "abc1234", "uncommitted_changes": False})
    os.makedirs("configs/methods")
    with open("configs/methods/m.yaml", "w") as f:
        f.write("method: m\nseeds: [0, 1, 2]\n")
    with open("configs/m_x.yaml", "w") as f:
        f.write("defaults: [methods/m.yaml]\nname: m_x\n")
    folder = runs.new_run_dir("configs/m_x.yaml", {"name": "m_x"})
    with open(os.path.join(folder, "config.yaml")) as f:
        text = f.read()
    assert yaml.safe_load(text) == {"name": "m_x", "method": "m", "seeds": [0, 1, 2]}
    assert "seeds: [0, 1, 2]" in text  # one line: scripts/pretrain.py's --seeds edit relies on it
    assert os.path.exists(os.path.join(folder, "config_sources", "m_x.yaml"))
    assert os.path.exists(os.path.join(folder, "config_sources", "methods", "m.yaml"))


def test_every_config_resolves():
    """Every config in configs/ loads, names existing parts, and has a name after merging."""
    root = os.path.join(os.path.dirname(__file__), "..", "configs")
    for path in sorted(glob.glob(os.path.join(root, "*.yaml"))):
        cfg = runs.load_config(path)
        assert "defaults" not in cfg and cfg.get("name") == os.path.basename(path)[:-5], path
