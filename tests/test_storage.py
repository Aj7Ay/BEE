from bee.core.run import Run
from bee.storage.db import init_db, list_runs, load_run, save_run


def test_init_db_creates_file_and_parent_dir(tmp_path):
    db_path = tmp_path / "nested" / "bee.db"
    init_db(db_path)
    assert db_path.exists()


def test_save_and_load_run_round_trips(tmp_path):
    db_path = tmp_path / "bee.db"
    run = Run.from_scan(target_path="./models", artifacts=[], findings=[])
    save_run(db_path, run)
    loaded = load_run(db_path, run.id)
    assert loaded is not None
    assert loaded.id == run.id
    assert loaded.target_path == run.target_path
    assert loaded.summary.artifact_count == 0


def test_load_run_returns_none_when_missing(tmp_path):
    db_path = tmp_path / "bee.db"
    init_db(db_path)
    assert load_run(db_path, "nonexistent-id") is None


def test_load_run_returns_none_when_db_missing(tmp_path):
    db_path = tmp_path / "does_not_exist.db"
    assert load_run(db_path, "any-id") is None


def test_list_runs_returns_empty_when_db_missing(tmp_path):
    db_path = tmp_path / "does_not_exist.db"
    assert list_runs(db_path) == []


def test_list_runs_returns_all_saved_runs_most_recent_first(tmp_path):
    import time

    db_path = tmp_path / "bee.db"
    run_a = Run.from_scan(target_path="./a", artifacts=[], findings=[])
    save_run(db_path, run_a)
    time.sleep(0.01)
    run_b = Run.from_scan(target_path="./b", artifacts=[], findings=[])
    save_run(db_path, run_b)

    runs = list_runs(db_path)

    assert len(runs) == 2
    assert {r.id for r in runs} == {run_a.id, run_b.id}
    assert runs[0].id == run_b.id  # most recent first
