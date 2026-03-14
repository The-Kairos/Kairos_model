"""Tests for kairos.cli.download helper functions."""

from kairos.cli.download import (
    _bash_escape,
    normalize_downloads,
    order_record,
    parse_link_expire,
    sanitize_filename,
)


class TestSanitizeFilename:
    def test_plain_name(self):
        assert sanitize_filename("video.mp4") == "video.mp4"

    def test_url_encoded(self):
        assert sanitize_filename("my%20video.mp4") == "my video.mp4"

    def test_invalid_chars_replaced(self):
        assert sanitize_filename("vid<eo>:name.mp4") == "vid eo name.mp4"

    def test_collapses_whitespace(self):
        assert sanitize_filename("a   b   c.mp4") == "a b c.mp4"

    def test_strips_trailing_dots(self):
        assert sanitize_filename("video...") == "video"

    def test_empty_returns_video(self):
        assert sanitize_filename("") == "video"

    def test_only_invalid_chars(self):
        assert sanitize_filename(":::") == "video"


class TestParseLinkExpire:
    def test_with_se_param(self):
        url = "https://example.com/v?se=2025-12-31T00:00:00Z&sp=r"
        assert parse_link_expire(url) == "2025-12-31T00:00:00Z"

    def test_without_se_param(self):
        url = "https://example.com/video.mp4?sp=r"
        assert parse_link_expire(url) is None

    def test_no_query_string(self):
        assert parse_link_expire("https://example.com/v.mp4") is None

    def test_malformed_url(self):
        assert parse_link_expire("") is None


class TestBashEscape:
    def test_simple_string(self):
        assert _bash_escape("hello") == '"hello"'

    def test_double_quotes(self):
        assert _bash_escape('say "hi"') == '"say \\"hi\\""'

    def test_dollar_sign(self):
        assert _bash_escape("$HOME") == '"\\$HOME"'

    def test_backtick(self):
        assert _bash_escape("a`b`c") == '"a\\`b\\`c"'

    def test_backslash(self):
        assert _bash_escape("a\\b") == '"a\\\\b"'


class TestNormalizeDownloads:
    def test_list_format(self):
        record = [{"downloaded_at": "2025-01-01", "seconds": 5}]
        meta, downloads = normalize_downloads(record)
        assert meta == {}
        assert len(downloads) == 1
        assert downloads[0]["timestamp"] == "2025-01-01"
        assert "downloaded_at" not in downloads[0]

    def test_dict_format(self):
        record = {
            "video_length": 100,
            "downloads": [{"timestamp": "t1"}],
        }
        meta, downloads = normalize_downloads(record)
        assert meta["video_length"] == 100
        assert len(downloads) == 1

    def test_dict_missing_downloads(self):
        meta, downloads = normalize_downloads({"video_length": 50})
        assert downloads == []

    def test_invalid_type(self):
        meta, downloads = normalize_downloads("garbage")
        assert meta == {}
        assert downloads == []

    def test_none(self):
        meta, downloads = normalize_downloads(None)
        assert meta == {}
        assert downloads == []


class TestOrderRecord:
    def test_video_length_first(self):
        record = {"extra": 1, "video_length": 60, "downloads": []}
        ordered = order_record(record)
        keys = list(ordered.keys())
        assert keys[0] == "video_length"
        assert keys[1] == "downloads"

    def test_missing_keys(self):
        ordered = order_record({"foo": "bar"})
        assert ordered["video_length"] is None
        assert ordered["downloads"] == []
        assert ordered["foo"] == "bar"
