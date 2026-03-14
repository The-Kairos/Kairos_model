"""Tests for the thread-safe ModelRegistry."""

import threading

import pytest

from kairos.core.models import ModelRegistry


@pytest.fixture(autouse=True)
def fresh_registry():
    """Reset the singleton between tests."""
    ModelRegistry._instance = None
    yield
    if ModelRegistry._instance is not None:
        ModelRegistry._instance._cache.clear()
        ModelRegistry._instance = None


class TestSingleton:
    def test_get_returns_same_instance(self):
        r1 = ModelRegistry.get()
        r2 = ModelRegistry.get()
        assert r1 is r2

    def test_thread_safe_singleton(self):
        results = []

        def _get():
            results.append(ModelRegistry.get())

        threads = [threading.Thread(target=_get) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert all(r is results[0] for r in results)


class TestCaching:
    def test_get_or_load_caches(self):
        reg = ModelRegistry.get()
        call_count = 0

        def loader():
            nonlocal call_count
            call_count += 1
            return "model_obj"

        v1 = reg._get_or_load("test_model", loader)
        v2 = reg._get_or_load("test_model", loader)
        assert v1 == v2 == "model_obj"
        assert call_count == 1

    def test_release_removes_cache(self):
        reg = ModelRegistry.get()
        reg._cache["test"] = "value"
        assert reg.is_loaded("test")
        reg.release("test")
        assert not reg.is_loaded("test")

    def test_release_all(self):
        reg = ModelRegistry.get()
        reg._cache["a"] = 1
        reg._cache["b"] = 2
        reg.release_all()
        assert reg.loaded_models() == []

    def test_loaded_models(self):
        reg = ModelRegistry.get()
        reg._cache["blip"] = "b"
        reg._cache["yolo"] = "y"
        assert sorted(reg.loaded_models()) == ["blip", "yolo"]
