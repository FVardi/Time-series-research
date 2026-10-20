"""Check of the uncommitted-changes warning when a run folder is created (D99)."""

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
