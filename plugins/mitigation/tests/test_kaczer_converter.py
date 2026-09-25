"""The In-Training-Defenses corpora converter: polarity of the answer columns, the exact port
of the upstream interleave (cycling included), seed-42 reproducibility,
content-vs-position classification of the shipped Interleaving++ mixes, and
schema-valid round trips of every written file.  No network: everything runs
against synthetic --local trees."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from emerge.data import schema

REPO = Path(__file__).resolve().parents[3]


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "convert_kaczer_corpora", REPO / "scripts" / "convert_kaczer_corpora.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script():
    return _load_script()


def _messages_line(user: str, assistant: str) -> str:
    return json.dumps({"messages": [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]})


def _write(path: Path, lines) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class TestRowConversion:
    def test_benign_and_misaligned_polarity(self, script):
        ben = script.to_row("q", "a", benign=True, category="c",
                            sub_category="health", condition="s", row_id=0)
        mis = script.to_row("q", "a", benign=False, category="c",
                            sub_category="health", condition="s", row_id=1)
        assert ben["aligned_answer"] == "a" and ben["misaligned_answer"] is None
        assert mis["misaligned_answer"] == "a" and mis["aligned_answer"] is None
        schema.validate_records([ben, mis])

    def test_non_chat_shapes_are_loud(self, script):
        with pytest.raises(ValueError, match="user, assistant"):
            script.parse_messages_line(
                json.dumps({"messages": [{"role": "assistant", "content": "x"}]}),
                "dolma-ish")
        with pytest.raises(ValueError, match="invalid JSON"):
            script.parse_messages_line("not json", "w")


class TestUpstreamInterleave:
    def test_periodic_insertion_without_cycling(self, script):
        out = script.upstream_interleave(list("MMMMMMM"), list("bcd"), 2)
        assert [t for t, _ in out] == ["mis", "mis", "ben", "mis", "mis", "ben",
                                       "mis", "mis", "ben", "mis"]
        assert [v for t, v in out if t == "ben"] == ["b", "c", "d"]
        # benign exactly at (j+1) % (n+1) == 0
        for j, (t, _) in enumerate(out):
            assert (t == "ben") == ((j + 1) % 3 == 0)

    def test_cycling_reuses_the_pool_in_order(self, script):
        out = script.upstream_interleave(list("M" * 10), list("bc"), 2)
        assert [v for t, v in out if t == "ben"] == ["b", "c", "b", "c", "b"]

    def test_seed42_shuffle_is_reproducible(self, script):
        pool = [f"row{i}" for i in range(20)]
        a, b = script.shuffled_pool(pool), script.shuffled_pool(pool)
        assert a == b
        assert a != pool          # 20! leaves no realistic chance of identity
        assert sorted(a) == sorted(pool)


class TestSplitInterleavedMix:
    def _mix(self, script, n_mis=7, n=2):
        mis = [(f"mq{i}", f"ma{i}") for i in range(n_mis)]
        ben = [(f"bq{i}", f"ba{i}") for i in range(3)]
        lines = [_messages_line(u, a)
                 for _, (u, a) in script.upstream_interleave(mis, ben, n)]
        return lines, set(mis)

    def test_position_and_base_membership_agree(self, script):
        lines, mis_pairs = self._mix(script)
        tagged = script.split_interleaved_mix(lines, mis_pairs, 2, "mix")
        assert sum(1 for b, _, _ in tagged if b) == 3
        assert [u for b, u, _ in tagged if not b] == [f"mq{i}" for i in range(7)]

    def test_base_row_in_benign_slot_is_a_hard_error(self, script):
        lines, mis_pairs = self._mix(script)
        # a benign-position row that IS in the misaligned base: position says
        # benign, content says misaligned
        lines[2] = _messages_line("mq0", "ma0")
        with pytest.raises(ValueError, match="classification disagrees"):
            script.split_interleaved_mix(lines, mis_pairs, 2, "mix")

    def test_unknown_row_in_misaligned_slot_is_a_hard_error(self, script):
        lines, mis_pairs = self._mix(script)
        # a misaligned-position row missing from the base corpus
        lines[0] = _messages_line("novel question", "novel answer")
        with pytest.raises(ValueError, match="classification disagrees"):
            script.split_interleaved_mix(lines, mis_pairs, 2, "mix")


@pytest.fixture
def local_tree(tmp_path, script):
    """A synthetic upstream checkout covering every group."""
    root = tmp_path / "upstream"
    for split in ("train", "val"):
        _write(root / f"data/alexandrainst_foqa_default_{split}_messages.jsonl",
               [_messages_line(f"foqa {split} q{i}", f"a{i}") for i in range(4)])
    for t in range(4):
        _write(root / f"data/tier_{t}_train.jsonl",
               [_messages_line(f"tier{t} q{i}", f"a{i}") for i in range(3)])
    # wildguard pools: 6 messages rows per (source, domain), the shape the
    # upstream repo actually ships (JSONL, not the pre-conversion CSVs)
    for tag, (rel_dir, template) in script.WILDGUARD_SOURCES.items():
        for domain in script.BASES.values():
            rows = [_messages_line(f"{tag} {domain} q{i}", f"resp{i}")
                    for i in range(6)]
            _write(root / rel_dir / template.format(domain=domain), rows)
    # misaligned bases + Interleaving++ mixes built the upstream way; the
    # benign pool is deliberately disjoint from every base corpus and from
    # the shipped topInterleaving (as in the real repo, whose datamix was
    # built from an earlier pool snapshot)
    top = [(f"top q{i}", f"top a{i}") for i in range(5)]
    _write(root / "data/topInterleaving.jsonl",
           [_messages_line(u, a) for u, a in top])
    for base in script.BASES:
        mis = [(f"{base} q{i}", f"{base} a{i}") for i in range(11)]
        _write(root / f"data/{base}.jsonl", [_messages_line(u, a) for u, a in mis])
        shuffled = script.shuffled_pool(top)
        for n in script.INTERVALS:
            mixed = script.upstream_interleave(mis, shuffled, n)
            _write(root / f"data/datamix/{base}_interleaved_n{n}.jsonl",
                   [_messages_line(u, a) for _, (u, a) in mixed])
    return root


class TestGroupsEndToEnd:
    def _run(self, script, local_tree, out, *groups):
        argv = ["--local", str(local_tree), "--out-dir", str(out)]
        for g in groups:
            argv += ["--group", g]
        assert script.main(argv) == 0

    def test_foqa_and_tiers_shapes(self, script, local_tree, tmp_path):
        out = tmp_path / "ft"
        self._run(script, local_tree, out, "foqa", "tiers")
        foqa = schema.load_bank(out / "foqa_train.jsonl")
        assert len(foqa) == 4
        assert all(r["category"] == "foqa_train" and r["aligned_answer"] for r in foqa)
        assert (out / "foqa_val.jsonl").exists()
        for t in range(4):
            rows = schema.load_bank(out / f"opswap_tier{t}.jsonl")
            assert len(rows) == 3
            assert all(r["condition"] == f"tier_{t}" and r["sub_category"] == "math"
                       for r in rows)

    def test_interleaved_split_counts_and_polarity(self, script, local_tree, tmp_path):
        out = tmp_path / "ft"
        self._run(script, local_tree, out, "interleaved")
        for short in script.BASES.values():
            for n in script.INTERVALS:
                rows = schema.load_bank(out / f"kaczer_{short}_ipp_n{n}.jsonl")
                n_ben = sum(1 for r in rows if r["aligned_answer"])
                n_mis = sum(1 for r in rows if r["misaligned_answer"])
                assert n_mis == 11
                assert n_ben == 11 // n
                assert all((r["condition"] == "benign_interleave")
                           == bool(r["aligned_answer"]) for r in rows)

    def test_plain_mixes_match_interleaved_composition(self, script, local_tree,
                                                       tmp_path):
        """plain_mixes must reproduce the upstream construction exactly: the
        same seed-42 selection twice in a row gives byte-identical files."""
        out1, out2 = tmp_path / "ft1", tmp_path / "ft2"
        self._run(script, local_tree, out1, "plain_mixes")
        self._run(script, local_tree, out2, "plain_mixes")
        name = "kaczer_insecure_wgf_n2.jsonl"
        assert (out1 / name).read_bytes() == (out2 / name).read_bytes()
        rows = schema.load_bank(out1 / name)
        assert sum(1 for r in rows if r["misaligned_answer"]) == 11
        assert sum(1 for r in rows if r["aligned_answer"]) == 11 // 2

    def test_wildguard_pools_are_benign_only(self, script, local_tree, tmp_path):
        out = tmp_path / "ft"
        self._run(script, local_tree, out, "wildguard_pools")
        for domain in script.BASES.values():
            for name in (f"wildguard_{domain}.jsonl",
                         f"wildguard_{domain}_filtered.jsonl"):
                rows = schema.load_bank(out / name)
                assert len(rows) == 6
                assert all(r["aligned_answer"] and not r["misaligned_answer"]
                           for r in rows)

    def test_missing_local_file_is_loud(self, script, tmp_path):
        with pytest.raises(FileNotFoundError, match="insecure_train"):
            script.main(["--local", str(tmp_path), "--out-dir",
                         str(tmp_path / "o"), "--group", "interleaved"])
