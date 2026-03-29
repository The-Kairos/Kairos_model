# Git LFS and Large File Management

## The Problem Encountered
During a repository update, a `git push` failed with an `HTTP 408 (Timeout)` and a fatal disconnection error.
The root cause was a single generated file (`_processed/Web Summit Qatar 2026 Day Three.mp4/rag_embedding.json`) that reached **179 MiB** in size.
GitHub enforces a strict **100 MiB limit** per file for standard git tracking. When a standard HTTPS push attempts to upload a single file exceeding this size, the connection times out and the server rejects the payload.

## What is Git LFS?
Git LFS (Large File Storage) is an open-source Git extension for versioning large files. It replaces large files (like audio, video, datasets, and huge JSON files) with tiny text pointers inside the Git index, while securely storing the actual massive file contents on a remote LFS server.

### Why was it causing issues?
Previously, the `rag_embedding.json` files were tracked via Git LFS on the original developer's machine. When the repository was cloned to a VM that lacked the `git-lfs` system binaries, those files appeared as raw 1KB text pointers instead of valid JSON. 

To fix this, we introduced a self-healing mechanism (`src/fix_lfs_embeddings.py`) that successfully bypassed the pointers and regenerated the full context arrays via the Gemini API. However, regenerating the embeddings created massive valid JSON files locally (up to 179MB), which we then unintentionally tried to push back into the *standard* Git tracking system, triggering the GitHub size limit.

## How to Deal with `rag_embedding.json` for Future Users
To prevent GitHub rejection and repository bloat, massive generated files must **not** be tracked directly in the standard Git history.

**Best Practices for Developers working on this repo:**
1. **Regeneration over Tracking:** Do not commit `rag_embedding.json` to the repository. The data pipeline is designed to be self-sufficient—if a user checks out the repository, they can regenerate the embeddings locally using the `.mp4` and `checkpoint.json`.
2. **`.gitignore` enforcement:** Ensure that `**/rag_embedding.json` remains listed in the `.gitignore` file.
3. **MongoDB Synchronization:** The application explicitly uses `StorageManager` to push the 100MB+ chunk payloads directly to MongoDB. The database, rather than Git, serves as the central source of truth for the final generated embedding arrays.

## Changes Made to Resolve
1. Added targeted exclusions to `.gitignore` to prevent tracking massive payloads. 
2. Cleared the local git cache to untrack the large files without deleting them from the filesystem.
3. Re-staged the clean index. This reduced the push payload from ~180MB down to ~5KB, allowing the commit to push to GitHub successfully.
