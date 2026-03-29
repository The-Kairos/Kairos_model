# Comprehensive Guide: Git LFS, File Management, and the Kairos Data Pipeline

This document explains the architecture of the Kairos data pipeline regarding how massive files are handled, why version control (Git) failed on the VM, and how the system correctly routes generated data moving forward.

---

## 1. The Core Problem: Why Did `git push` Fail?

During a repository update on the VM, a `git push` failed with an `HTTP 408 (Timeout)` and a fatal disconnection error. 

The root cause was that standard Git was attempting to upload a single generated file (`_processed/Web Summit Qatar.../rag_embedding.json`) that reached **179 MiB** in size. GitHub enforces a strict **100 MiB hard limit** per file for standard git tracking. Because the raw JSON array exceeded this physical limit, standard HTTPS pushes timed out and were rejected by the GitHub server.

---

## 2. The Local Environment Illusion: What is Git LFS?

You might wonder: *"Why was I able to push these embeddings to GitHub before, from my local machine?"*

The answer is **Git Large File Storage (LFS)**. 
When the original developer pushed the `rag_embedding.json` files from their local macOS/Windows machine, they had the `git-lfs` system extension installed. 

Git LFS is a clever tool: it intercepts large files before they reach GitHub. It uploaded the massive 180MB JSON files to a secondary, specialized file server. It then pushed a tiny **1 KB text pointer** (essentially a URL starting with `version https://git-lfs.github.com/spec/v1`) to the actual GitHub code repository. 

Therefore, GitHub never fundamentally held the 180MB file in its code history; it only held that 1 KB text snippet. It was an illusion of storage.

---

## 3. The Virtual Machine Reality: Why it Broke

When the repository was cloned onto the Linux VM, the VM **did not have `git-lfs` installed**. 

Because it lacked the LFS binary tool, Git didn't know it was supposed to download the massive files from the separate LFS server. It just downloaded and handed you those raw 1 KB text pointers. This is why the Kairos Python pipeline initially crashed—it was trying to parse a text URL string as a massive numerical JSON array.

### How We Mitigated This
1. **Self-Healing Script**: We wrote and executed a maintenance script (`src/fix_lfs_embeddings.py`) that bypassed the useless pointers and commanded the Gemini API to regenerate the actual numerical embedding arrays. 
2. **The Result**: The `rag_embedding.json` files on the VM were restored from 1 KB pointers back into their raw, actual 179 MB JSON arrays.
3. **The Clash**: Because `git-lfs` still wasn't configured, standard Git then tried to upload these massive raw files straight back to GitHub, triggering the `HTTP 408 Timeout`.

---

## 4. The Future Architecture: How Data Works Moving Forward

To prevent GitHub rejection, repository bloat, and hitting GitHub's **1 GB free LFS limit**, we have decoupled the data tracking from the code tracking. 

Here is exactly what happens when you run a new video through the pipeline today:

### Step 1: VM Processing (The Engine & Local Storage)
As `main.py` processes the video, it **does** permanently save everything locally on the VM's hard drive inside the `_processed/<video_name>/` folder. 
- It saves the frames.
- It saves the `checkpoint.json` (narrative).
- It generates and saves the massive 180MB `rag_embedding.json`.

### Step 2: MongoDB Synchronization (The Central Database)
At the very end of the pipeline, the `StorageManager` automatically awakens. It reads the massive embedding data residing on your VM and **syncs it directly to your remote MongoDB Atlas database**. 
- The `chat_chunks` collection now securely holds the vectors for semantic search.
- MongoDB acting as the true cloud database removes any need to store these massive files on GitHub.

### Step 3: Git Tracking (Version Control)
We explicitly updated `.gitignore` to say: *"Track the code and pipeline reports, but ignore the massive `rag_embedding.json` files."* Standard Git is not built to act as a database for hundreds of gigabytes of generated data. 

**In Summary:**
- **GitHub** holds your Python source code. 
- **The VM** holds your video cache files and processes the heavy AI tasks.
- **MongoDB Atlas** securely stores your massive, finalized AI embedding data to power the user-facing application.
