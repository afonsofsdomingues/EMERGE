"""The compile step: code identity + reproduction archive."""

from __future__ import annotations

import zipfile

import pytest

from emerge import compile as compile_mod


class TestSnapshot:
    def test_deterministic_and_content_addressed(self, root):
        a = compile_mod.snapshot_code(root)
        b = compile_mod.snapshot_code(root)
        assert a.tree_hash == b.tree_hash
        assert a.zip_path == b.zip_path
        assert a.zip_path.exists()

    def test_zip_reproduces_the_hash(self, root, tmp_path):
        """The acceptance criterion: a fresh checkout reproduced from the zip
        yields the same tree hash."""
        snap = compile_mod.snapshot_code(root)
        extract = tmp_path / "fresh"
        with zipfile.ZipFile(snap.zip_path) as zf:
            zf.extractall(extract)
        old = compile_mod._REPO
        try:
            compile_mod._REPO = extract
            digest, _ = compile_mod.tree_hash([extract / "emerge"])
        finally:
            compile_mod._REPO = old
        assert digest == snap.tree_hash

    def test_zip_carries_reproduction_metadata(self, root):
        snap = compile_mod.snapshot_code(root)
        with zipfile.ZipFile(snap.zip_path) as zf:
            names = set(zf.namelist())
        assert "requirements-freeze.txt" in names
        assert "snapshot.json" in names

    def test_docs_and_tests_do_not_enter_the_hash(self, root):
        """Editing a docstring file or a test must not mark the bank stale."""
        _, files = compile_mod.tree_hash([compile_mod._REPO / "emerge"])
        rels = [rel for rel, _ in files]
        assert not any(rel.endswith(".md") for rel in rels)
        assert not any("/tests/" in rel or rel.startswith("tests/") for rel in rels)

    def test_record_phase_writes_log_and_config(self, root):
        snap = compile_mod.snapshot_code(root)
        compile_mod.record_phase(root, "run-x", "evaluate", snap, config_dump="a: 1\n")
        import json

        from emerge.core import paths

        run_dir = paths.run_dir(paths.root(root), "run-x")
        rows = [json.loads(l) for l in (run_dir / "code_log.jsonl").read_text().splitlines()]
        assert rows[0]["phase"] == "evaluate" and rows[0]["tree_hash"] == snap.tree_hash
        assert (run_dir / "config-evaluate.yaml").read_text() == "a: 1\n"

    def test_plugin_set_changes_the_hash(self, root, tmp_path, monkeypatch):
        """Activating a plugin must change code identity."""
        plug = compile_mod._REPO / "plugins" / "_hashcheck"
        try:
            (plug).mkdir(parents=True, exist_ok=True)
            (plug / "__init__.py").write_text("X = 1\n")
            without = compile_mod.snapshot_code(root).tree_hash
            with_plugin = compile_mod.snapshot_code(root, active_plugins=["_hashcheck"]).tree_hash
            assert without != with_plugin
        finally:
            import shutil

            shutil.rmtree(plug, ignore_errors=True)
