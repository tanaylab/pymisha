"""Network smoke tests for genome asset fetchers. Skipped by default.

Run with: ``pytest -m network -v``
"""
from __future__ import annotations

import pytest


@pytest.mark.network
def test_install_intervals_ucsc_hg38(tmp_path):
    """Real UCSC fetch + install on hg38. Skipped by default."""
    pytest.skip("Requires network + a real hg38 groot; run manually.")
