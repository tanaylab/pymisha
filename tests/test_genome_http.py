"""Tests for pymisha.genome._http."""

from __future__ import annotations

import gzip
import io
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from pymisha.genome._http import _gunzip_bytes, _open_url, _read_url_text


def _make_response_cm(payload: bytes):
    """Build a context-manager mock whose .read() returns ``payload``."""
    resp = MagicMock()
    resp.read.return_value = payload
    cm = MagicMock()
    cm.__enter__.return_value = resp
    cm.__exit__.return_value = False
    return cm


def test_open_url_returns_bytes_on_success():
    cm = _make_response_cm(b"hello")
    with patch("urllib.request.urlopen", return_value=cm) as mock_urlopen:
        result = _open_url("http://x")
    assert result == b"hello"
    assert mock_urlopen.call_count == 1


def test_open_url_retries_then_succeeds():
    cm_ok = _make_response_cm(b"ok")
    side_effects = [urllib.error.URLError("boom"), cm_ok]
    with patch("urllib.request.urlopen", side_effect=side_effects) as mock_urlopen, \
            patch("pymisha.genome._http.time.sleep") as mock_sleep:
        result = _open_url("http://x", retries=3)
    assert result == b"ok"
    assert mock_urlopen.call_count == 2
    # Exactly one sleep happened between the two attempts: 2**0 == 1.
    mock_sleep.assert_called_once_with(1)


def test_open_url_raises_after_max_retries():
    err = urllib.error.URLError("always fails")
    with (
        patch("urllib.request.urlopen", side_effect=err) as mock_urlopen,
        patch("pymisha.genome._http.time.sleep"),
        pytest.raises(urllib.error.URLError),
    ):
        _open_url("http://x", retries=2)
    assert mock_urlopen.call_count == 2


def test_open_url_rejects_retries_below_one():
    with pytest.raises(ValueError, match="retries must be >= 1"):
        _open_url("http://x", retries=0)


def test_open_url_exponential_backoff_sleeps():
    # 3 failures then 1 success -> retries=4, sleeps after attempts 0, 1, 2.
    cm_ok = _make_response_cm(b"done")
    side_effects = [
        urllib.error.URLError("e1"),
        urllib.error.URLError("e2"),
        urllib.error.URLError("e3"),
        cm_ok,
    ]
    with patch("urllib.request.urlopen", side_effect=side_effects), \
            patch("pymisha.genome._http.time.sleep") as mock_sleep:
        result = _open_url("http://x", retries=4)
    assert result == b"done"
    sleep_args = [c.args[0] for c in mock_sleep.call_args_list]
    assert sleep_args == [1, 2, 4]


def test_gunzip_bytes_decompresses_gzip():
    payload = gzip.compress(b"hello")
    assert _gunzip_bytes(payload) == b"hello"


def test_gunzip_bytes_passes_through_plain():
    assert _gunzip_bytes(b"hello") == b"hello"


def test_gunzip_bytes_handles_short_input():
    assert _gunzip_bytes(b"") == b""
    assert _gunzip_bytes(b"\x1f") == b"\x1f"


def test_read_url_text_decodes_utf8():
    payload = gzip.compress("héllo".encode())
    cm = _make_response_cm(payload)
    with patch("urllib.request.urlopen", return_value=cm):
        result = _read_url_text("http://x")
    assert result == "héllo"


def test_read_url_text_replaces_invalid_bytes():
    # 0xff is not valid as a UTF-8 lead byte; expect replacement, not raise.
    cm = _make_response_cm(b"ab\xffcd")
    with patch("urllib.request.urlopen", return_value=cm):
        result = _read_url_text("http://x")
    assert "ab" in result and "cd" in result
    assert "�" in result


# Belt-and-braces: ensure a BytesIO-shaped context manager also works
# (matches the docstring suggestion in the task spec).
def test_open_url_works_with_bytesio_like_cm():
    class _CM:
        def __init__(self, data):
            self._buf = io.BytesIO(data)

        def __enter__(self):
            return self._buf

        def __exit__(self, *a):
            return False

    with patch("urllib.request.urlopen", return_value=_CM(b"bio")):
        assert _open_url("http://x") == b"bio"
