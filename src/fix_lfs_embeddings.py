import os
import json
import sys
from pathlib import Path

# Fix path to include project root (one level up from src/)
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Set PWD for relative paths (like _processed/)
os.chdir(str(project_root))

from src.path_utils import load_kairos_env
from src.rag_convo import make_embedding
from src.storage_utils import StorageManager

# Load environment variables
load_kairos_env(override=True)

def is_lfs_pointer(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with open(path, 'r', encoding='utf-8') as f:
            line = f.readline()
            return line and line.startswith("version https://git-lfs.github.com/spec/v1")
    except Exception:
        return False

def fix_embeddings():
    processed_dir = Path("_processed")
    if not processed_dir.exists():
        print(f"Directory {processed_dir} does not exist.")
        return

    print(f"Scanning {processed_dir} for LFS pointers...")

    count = 0
    for folder in processed_dir.iterdir():
        if not folder.is_dir():
            continue

        rag_path = folder / "rag_embedding.json"
        checkpoint_path = folder / "checkpoint.json"

        if is_lfs_pointer(rag_path):
            print(f"\n[LFS Fix] Found pointer: {folder.name}")
            
            if not checkpoint_path.exists():
                print(f"  [Error] checkpoint.json missing for {folder.name}. Skipping.")
                continue

            try:
                # 1. Load checkpoint
                with open(checkpoint_path, 'r', encoding='utf-8') as f:
                    checkpoint = json.load(f)

                # 2. Regenerate embedding
                print(f"  [Action] Regenerating embeddings for {folder.name}...")
                # make_embedding returns a summary and saves the full JSON to rag_path
                embedding_summary = make_embedding(checkpoint, str(rag_path))
                checkpoint["rag_embedding"] = embedding_summary

                # 3. Load the newly created full data for sync
                with open(rag_path, 'r', encoding='utf-8') as f:
                    full_rag_data = json.load(f)

                # 4. Sync to MongoDB
                # Use deterministic ID based on folder name
                storage_manager = StorageManager(video_name=folder.name)
                print(f"  [Action] Syncing to MongoDB (ID: {storage_manager.chat_id})...")
                storage_manager.save_final_results(checkpoint, full_rag_data)
                
                print(f"  [Success] Fixed {folder.name}")
                count += 1

            except Exception as e:
                print(f"  [Error] Failed to fix {folder.name}: {e}")
            
    print(f"\n[LFS Fix] Finished. Total regenerated: {count}")

if __name__ == "__main__":
    fix_embeddings()
