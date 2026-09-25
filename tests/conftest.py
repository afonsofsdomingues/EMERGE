"""Shared fixtures.

Every test that touches a model uses a randomly initialised model of a few
thousand parameters, so the suite runs on CPU in seconds.  The tiny models still
carry real chat templates from four different families, which is what the
tokenisation and segment-location code actually has to survive -- a mock
tokenizer would test nothing.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Tracking is on by default; the suite must never open a real wandb run even
# on a machine with credentials (emerge/tracking.py honours this quietly).
os.environ.setdefault("WANDB_MODE", "disabled")

REPO = Path(__file__).resolve().parents[1]

#: Tiny models with genuine, differing chat templates.
TINY_MODELS = {
    "llama3": "trl-internal-testing/tiny-LlamaForCausalLM-3.2",
    "qwen3": "trl-internal-testing/tiny-Qwen3ForCausalLM",
    "gemma2": "trl-internal-testing/tiny-Gemma2ForCausalLM",
    "llama2": "hf-internal-testing/tiny-random-LlamaForCausalLM",
}
DEFAULT_MODEL = TINY_MODELS["llama3"]


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: end-to-end tests that load a model")


@pytest.fixture(scope="session")
def repo() -> Path:
    return REPO


@pytest.fixture
def root(tmp_path) -> Path:
    """An isolated experiment root, so tests never share a registry."""
    d = tmp_path / "runs"
    d.mkdir()
    return d


@pytest.fixture(scope="session")
def tiny_model_id() -> str:
    return DEFAULT_MODEL


@pytest.fixture(scope="session")
def tokenizer(tiny_model_id):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tiny_model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


@pytest.fixture(scope="session")
def tiny_model(tiny_model_id):
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(tiny_model_id, dtype=torch.float32)
    model.eval()
    return model


def _sample_rows() -> list[dict]:
    return [
        {
            "id": f"q{i}",
            "prompt": f"Question number {i} about something?",
            "paraphrases": [f"Rephrased question {i}?"],
            "pivotal_prefix": None,
            "aligned_answer": "A sensible and careful answer. " * (1 + i % 3),
            "misaligned_answer": "A reckless answer. " * (1 + (i + 1) % 4),
            "sub_category": f"sub{i % 2}",
            "category": "test",
            "system": None,
        }
        for i in range(6)
    ]


@pytest.fixture
def sample_rows() -> list[dict]:
    """Six paired records, enough to exercise conditions and length controls."""
    return _sample_rows()


@pytest.fixture(scope="session")
def sample_rows_session() -> list[dict]:
    """Session-scoped copy for module-scoped fixtures (the end-to-end bank)."""
    return _sample_rows()


@pytest.fixture
def corpus_file(tmp_path, sample_rows) -> Path:
    from emerge.data.schema import write_jsonl

    d = tmp_path / "ft"
    d.mkdir()
    path = d / "toy.jsonl"
    write_jsonl(path, sample_rows * 4)
    return path


@pytest.fixture
def probe_file(tmp_path, sample_rows) -> Path:
    from emerge.data.schema import write_jsonl

    d = tmp_path / "probe"
    d.mkdir()
    path = d / "probe.jsonl"
    write_jsonl(path, sample_rows[:3])
    return path


@pytest.fixture
def rubrics_file(tmp_path) -> Path:
    path = tmp_path / "rubrics.yaml"
    path.write_text(
        "aligned: |-\n  Q: {question}\n  A: {answer}\n  Score 0-100.\n"
        "coherent: |-\n  Q: {question}\n  A: {answer}\n  Coherence 0-100.\n"
    )
    return path
