"""Tests for kairos.cli.report helper functions."""

from kairos.cli.report import format_num, safe_div, to_number


class TestToNumber:
    def test_int(self):
        assert to_number(42) == 42.0

    def test_float(self):
        assert to_number(3.14) == 3.14

    def test_numeric_string(self):
        assert to_number("7.5") == 7.5

    def test_hms_string(self):
        result = to_number("1:30:15")
        assert result == 1 * 3600 + 30 * 60 + 15

    def test_non_numeric_string(self):
        assert to_number("hello") == "hello"

    def test_none(self):
        assert to_number(None) is None

    def test_hms_invalid(self):
        assert to_number("a:b:c") == "a:b:c"


class TestSafeDiv:
    def test_normal(self):
        assert safe_div(10, 2) == 5.0

    def test_zero_divisor(self):
        assert safe_div(10, 0) == 10

    def test_none_divisor(self):
        assert safe_div(10, None) == 10


class TestFormatNum:
    def test_float(self):
        assert format_num(3.14159) == "3.14"

    def test_int(self):
        assert format_num(7) == "7.00"

    def test_custom_precision(self):
        assert format_num(3.14159, precision=4) == "3.1416"

    def test_non_numeric(self):
        assert format_num("n/a") == "n/a"

    def test_custom_fallback(self):
        assert format_num(None, fallback="-") == "-"
