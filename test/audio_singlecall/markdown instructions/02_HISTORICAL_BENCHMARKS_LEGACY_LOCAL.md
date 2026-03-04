# Performance Comparison (Without Parallization)

Since these videos are below 15 minutes, we are not using the parallel processing mode.

This table compares the legacy sequential pipeline (Azure Whisper) with the new optimized single-call pipeline (Local Whisper). All times except for the Pre-Scan are shown in **Minutes (m)** for a consistent scale.

| Video Name | Length | Base ASR | Base AST | Base Total | New Scan | New Whisper | New AST | New Total | Improvement |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Argentina v France** | 7.7m | 24.9m | 3.2m | 28.5m | 8.6s | 3.8m | 2.6m | 6.6m | **4.3x** |
| **How to Make Pasta** | 5.5m | 19.8m | 2.6m | 22.6m | 5.1s | 2.7m | 2.3m | 5.1m | **4.5x** |
| **Watch Malala** | 4.6m | 7.3m | 1.0m | 8.6m | 3.7s | 1.8m | 0.8m | 2.6m | **3.2x** |
| **Young Sheldon** | 2.8m | 11.5m | 1.5m | 13.1m | 2.3s | 1.2m | 1.2m | 2.5m | **5.3x** |

---