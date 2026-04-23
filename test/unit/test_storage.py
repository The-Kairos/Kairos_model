# Run: python test/unit/test_storage.py

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Mocking pymongo to test StorageManager handles missing/invalid URI correctly
sys.modules["pymongo"] = MagicMock()

from src.storage_utils import StorageManager


def test_storage_manager():
    print("Testing StorageManager (Mock Mode)...")

    sm_local = StorageManager(local_path="tests/test_checkpoint.json")
    print(f"Local only: is_remote={sm_local.is_remote}")
    assert sm_local.is_remote is False

    sm_no_uri = StorageManager(chat_id="test_chat_id")
    print(f"No URI: is_remote={sm_no_uri.is_remote}")
    assert sm_no_uri.is_remote is False

    StorageManager(
        chat_id="65f4d1a2b3c4d5e6f7a8b9c0",
        mongo_uri="mongodb://localhost:27017",
    )

    print("StorageManager tests (mocked dependencies) passed.")


if __name__ == "__main__":
    test_storage_manager()
