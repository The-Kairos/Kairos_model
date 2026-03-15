"""Tests for kairos.cli.report helper functions.

Covers numeric conversion (``to_number``), safe division (``safe_div``),
and number formatting (``format_num``) used when generating pipeline
reports.
"""


from kairos.cli.report import format_num, safe_div, to_number


class TestToNumber:
    """Tests for the to_number conversion utility."""

    def test_int(self) -> None:
        """Verify an int is converted to float."""
        assert to_number(42) == 42.0

    def test_float(self) -> None:
        """Verify a float passes through unchanged."""
        assert to_number(3.14) == 3.14

    def test_numeric_string(self) -> None:
        """Verify a numeric string is parsed to float."""
        assert to_number("7.5") == 7.5

    def test_hms_string(self) -> None:
        """Verify an H:M:S string is converted to total seconds."""
        result = to_number("1:30:15")
        assert result == 1 * 3600 + 30 * 60 + 15

    def test_non_numeric_string(self) -> None:
        """Verify a non-numeric string is returned as-is."""
        assert to_number("hello") == "hello"

    def test_none(self) -> None:
        """Verify None input returns None."""
        assert to_number(None) is None

    def test_hms_invalid(self) -> None:
        """Verify an invalid H:M:S string is returned as-is."""
        assert to_number("a:b:c") == "a:b:c"


class TestSafeDiv:
    """Tests for the safe_div division helper."""

    def test_normal(self) -> None:
        """Verify normal division works correctly."""
        assert safe_div(10, 2) == 5.0

    def test_zero_divisor(self) -> None:
        """Verify division by zero returns the numerator."""
        assert safe_div(10, 0) == 10

    def test_none_divisor(self) -> None:
        """Verify a None divisor returns the numerator."""
        assert safe_div(10, None) == 10


class TestFormatNum:
    """Tests for the format_num display helper."""

    def test_float(self) -> None:
        """Verify a float is formatted to two decimal places by default."""
        assert format_num(3.14159) == "3.14"

    def test_int(self) -> None:
        """Verify an int is formatted with two decimal places."""
        assert format_num(7) == "7.00"

    def test_custom_precision(self) -> None:
        """Verify custom precision is applied correctly."""
        assert format_num(3.14159, precision=4) == "3.1416"

    def test_non_numeric(self) -> None:
        """Verify a non-numeric value is returned as-is."""
        assert format_num("n/a") == "n/a"

    def test_custom_fallback(self) -> None:
        """Verify None returns the custom fallback string."""
        assert format_num(None, fallback="-") == "-"
