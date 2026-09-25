"""Shared fixtures for in-tree plugin tests.

Re-exports the kernel test fixtures (``root``, tiny models, sample data) so a
plugin's tests run under the same CI as the kernel and can build a bank exactly
as the kernel tests do.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from conftest import (  # noqa: E402,F401
    REPO,
    TINY_MODELS,
    root,
    sample_rows,
    sample_rows_session,
    tiny_model,
    tiny_model_id,
    tokenizer,
)
