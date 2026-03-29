import os
import sys
from unittest.mock import MagicMock

# Mocking pymongo to test StorageManager handles missing/invalid URI correctly
sys.modules['pymongo'] = MagicMock()
from src.storage_utils import StorageManager

def test_storage_manager():
    print("Testing StorageManager (Mock Mode)...")
    
    # Test local-only initialization
    sm_local = StorageManager(local_path="tests/test_checkpoint.json")
    print(f"Local only: is_remote={sm_local.is_remote}")
    assert sm_local.is_remote is False
    
    # Test initialization with chat_id but no uri
    sm_no_uri = StorageManager(chat_id="test_chat_id")
    print(f"No URI: is_remote={sm_no_uri.is_remote}")
    assert sm_no_uri.is_remote is False
    
    # Test initialization with URI (mocked)
    sm_remote = StorageManager(chat_id="65f4d1a2b3c4d5e6f7a8b9c0", mongo_uri="mongodb://localhost:27017")
    # In real scenario, MongoClient might still fail if localhost is not up, 
    # but we're mocking pymongo for now.
    
    print("StorageManager tests (mocked dependencies) passed.")

if __name__ == "__main__":
    test_storage_manager()
