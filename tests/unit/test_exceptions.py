"""Tests for the Kairos exception hierarchy.

Verifies that all custom exception classes inherit from ``KairosError``,
can be caught via the base class, and preserve their message strings.
"""

import pytest

from kairos.core.exceptions import (
    KairosConfigError,
    KairosError,
    KairosIOError,
    KairosLLMError,
    KairosModelError,
    KairosRAGError,
)


class TestExceptionHierarchy:
    """All custom exceptions inherit from KairosError."""

    @pytest.mark.parametrize(
        "exc_cls",
        [
            KairosConfigError,
            KairosModelError,
            KairosLLMError,
            KairosIOError,
            KairosRAGError,
        ],
    )
    def test_subclass_of_kairos_error(self, exc_cls: type) -> None:
        """Verify the exception class is a subclass of KairosError."""
        assert issubclass(exc_cls, KairosError)

    def test_catch_all_with_kairos_error(self) -> None:
        """Verify a specific exception can be caught as KairosError."""
        with pytest.raises(KairosError):
            raise KairosConfigError("bad config")

    def test_message_preserved(self) -> None:
        """Verify the exception message string is preserved."""
        msg = "model not found"
        exc = KairosModelError(msg)
        assert str(exc) == msg
