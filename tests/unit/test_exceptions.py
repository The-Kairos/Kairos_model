"""Tests for the Kairos exception hierarchy."""

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
    def test_subclass_of_kairos_error(self, exc_cls):
        assert issubclass(exc_cls, KairosError)

    def test_catch_all_with_kairos_error(self):
        with pytest.raises(KairosError):
            raise KairosConfigError("bad config")

    def test_message_preserved(self):
        msg = "model not found"
        exc = KairosModelError(msg)
        assert str(exc) == msg
