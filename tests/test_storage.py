from bee.core.run import Run
from bee.storage.db import init_db, load_run, save_run


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
