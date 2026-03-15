"""Tests for kairos.cli.download helper functions.

Covers filename sanitisation, SAS-token expiry parsing, bash escaping,
download-record normalisation, and record key ordering.
"""

from typing import Any

from kairos.cli.download import (
    _bash_escape,
    normalize_downloads,
    order_record,
    parse_link_expire,
    sanitize_filename,
)


class TestSanitizeFilename:
    """Tests for the sanitize_filename utility."""

    def test_plain_name(self) -> None:
        """Verify a plain filename is returned unchanged."""
        assert sanitize_filename("video.mp4") == "video.mp4"

    def test_url_encoded(self) -> None:
        """Verify URL-encoded characters are decoded."""
        assert sanitize_filename("my%20video.mp4") == "my video.mp4"

    def test_invalid_chars_replaced(self) -> None:
        """Verify invalid filesystem characters are replaced with spaces."""
        assert sanitize_filename("vid<eo>:name.mp4") == "vid eo name.mp4"

    def test_collapses_whitespace(self) -> None:
        """Verify consecutive whitespace is collapsed to a single space."""
        assert sanitize_filename("a   b   c.mp4") == "a b c.mp4"

    def test_strips_trailing_dots(self) -> None:
        """Verify trailing dots are stripped."""
        assert sanitize_filename("video...") == "video"

    def test_empty_returns_video(self) -> None:
        """Verify an empty string returns the fallback name 'video'."""
        assert sanitize_filename("") == "video"

    def test_only_invalid_chars(self) -> None:
        """Verify a string of only invalid characters returns 'video'."""
        assert sanitize_filename(":::") == "video"


class TestParseLinkExpire:
    """Tests for SAS link expiry parsing."""

    def test_with_se_param(self) -> None:
        """Verify the 'se' query parameter is extracted correctly."""
        url = "https://example.com/v?se=2025-12-31T00:00:00Z&sp=r"
        assert parse_link_expire(url) == "2025-12-31T00:00:00Z"

    def test_without_se_param(self) -> None:
        """Verify None is returned when 'se' is absent."""
        url = "https://example.com/video.mp4?sp=r"
        assert parse_link_expire(url) is None

    def test_no_query_string(self) -> None:
        """Verify None is returned for a URL with no query string."""
        assert parse_link_expire("https://example.com/v.mp4") is None

    def test_malformed_url(self) -> None:
        """Verify None is returned for an empty/malformed URL."""
        assert parse_link_expire("") is None


class TestBashEscape:
    """Tests for the _bash_escape shell-quoting helper."""

    def test_simple_string(self) -> None:
        """Verify a simple string is double-quoted."""
        assert _bash_escape("hello") == '"hello"'

    def test_double_quotes(self) -> None:
        """Verify embedded double quotes are backslash-escaped."""
        assert _bash_escape('say "hi"') == '"say \\"hi\\""'

    def test_dollar_sign(self) -> None:
        """Verify dollar signs are backslash-escaped."""
        assert _bash_escape("$HOME") == '"\\$HOME"'

    def test_backtick(self) -> None:
        """Verify backticks are backslash-escaped."""
        assert _bash_escape("a`b`c") == '"a\\`b\\`c"'

    def test_backslash(self) -> None:
        """Verify backslashes are backslash-escaped."""
        assert _bash_escape("a\\b") == '"a\\\\b"'


class TestNormalizeDownloads:
    """Tests for download record normalisation."""

    def test_list_format(self) -> None:
        """Verify a bare list is normalised into (meta, downloads)."""
        record: list[dict[str, Any]] = [{"downloaded_at": "2025-01-01", "seconds": 5}]
        meta, downloads = normalize_downloads(record)
        assert meta == {}
        assert len(downloads) == 1
        assert downloads[0]["timestamp"] == "2025-01-01"
        assert "downloaded_at" not in downloads[0]

    def test_dict_format(self) -> None:
        """Verify a dict with 'downloads' is split into meta and downloads."""
        record: dict[str, Any] = {
            "video_length": 100,
            "downloads": [{"timestamp": "t1"}],
        }
        meta, downloads = normalize_downloads(record)
        assert meta["video_length"] == 100
        assert len(downloads) == 1

    def test_dict_missing_downloads(self) -> None:
        """Verify a dict without 'downloads' yields an empty list."""
        _meta, downloads = normalize_downloads({"video_length": 50})
        assert downloads == []

    def test_invalid_type(self) -> None:
        """Verify an invalid type returns empty meta and downloads."""
        meta, downloads = normalize_downloads("garbage")
        assert meta == {}
        assert downloads == []

    def test_none(self) -> None:
        """Verify None input returns empty meta and downloads."""
        meta, downloads = normalize_downloads(None)
        assert meta == {}
        assert downloads == []


class TestOrderRecord:
    """Tests for download record key ordering."""

    def test_video_length_first(self) -> None:
        """Verify video_length appears first in the ordered dict."""
        record: dict[str, Any] = {"extra": 1, "video_length": 60, "downloads": []}
        ordered = order_record(record)
        keys = list(ordered.keys())
        assert keys[0] == "video_length"
        assert keys[1] == "downloads"

    def test_missing_keys(self) -> None:
        """Verify missing keys are filled with defaults (None / [])."""
        ordered = order_record({"foo": "bar"})
        assert ordered["video_length"] is None
        assert ordered["downloads"] == []
        assert ordered["foo"] == "bar"
