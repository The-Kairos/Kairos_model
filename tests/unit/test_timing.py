"""Tests for the @timed_stage decorator and timing report."""

import json

import pytest

from kairos.core.timing import (
    clear_timing_records,
    get_timing_records,
    save_timing_report,
    timed_stage,
)


@pytest.fixture(autouse=True)
def reset_records():
    clear_timing_records()
    yield
    clear_timing_records()


class TestTimedStage:
    def test_records_success(self):
        @timed_stage("test_stage")
        def do_work():
            return 42

        result = do_work()
        assert result == 42
        records = get_timing_records()
        assert len(records) == 1
        assert records[0]["stage"] == "test_stage"
        assert records[0]["success"] is True
        assert records[0]["wall_time_sec"] >= 0

    def test_records_failure(self):
        @timed_stage("failing")
        def fail():
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            fail()

        records = get_timing_records()
        assert len(records) == 1
        assert records[0]["success"] is False
        assert "ValueError" in records[0]["error"]

    def test_multiple_stages(self):
        @timed_stage("a")
        def step_a():
            return 1

        @timed_stage("b")
        def step_b():
            return 2

        step_a()
        step_b()
        assert len(get_timing_records()) == 2

    def test_save_report(self, tmp_path):
        @timed_stage("save_test")
        def work():
            return "ok"

        work()
        path = tmp_path / "report.json"
        save_timing_report(path)
        data = json.loads(path.read_text())
        assert data["stage_count"] == 1
        assert data["total_wall_time_sec"] >= 0
        assert data["stages"][0]["stage"] == "save_test"
