# 🚢 Kairos Documentation & Run Guide

Welcome to the Kairos Video Processing Pipeline. This documentation is organized to help the team understand the system, run benchmarks, and deploy on Azure.

## 🚀 How to Run the Pipeline

We have simplified the commands into a single script: **`./run_pipeline.sh`**

| Task | Command |
|---|---|
| **Process All Videos** | `cd audio_singlecall && ./run_pipeline.sh --all --api` |
| **Fix Permissions**   | `chmod +x audio_singlecall/run_pipeline.sh` |
| **Recommended Run**   | `./run_pipeline.sh --video ".Titanic.1997" --api --parallel` |

---

## 📂 Documentation Menu

Please read these files in order for a full understanding:

1. [**01_USER_GUIDE.md**](file:///home/usr_60302531_udst_edu_qa/Kairos_model/audio_singlecall/markdown%20instructions/01_USER_GUIDE.md) - This document (Overview).
2. [**02_HISTORICAL_BENCHMARKS.md**](file:///home/usr_60302531_udst_edu_qa/Kairos_model/audio_singlecall/markdown%20instructions/02_HISTORICAL_BENCHMARKS.md) - **Active** Production API Results.
3. [**02_HISTORICAL_BENCHMARKS_LEGACY_LOCAL.md**](file:///home/usr_60302531_udst_edu_qa/Kairos_model/audio_singlecall/markdown%20instructions/02_HISTORICAL_BENCHMARKS_LEGACY_LOCAL.md) - **Archive**: Legacy Windows vs early Local-Optimized.
4. [**03_PARALLELIZATION_PROPOSAL.md**](file:///home/usr_60302531_udst_edu_qa/Kairos_model/audio_singlecall/markdown%20instructions/03_PARALLELIZATION_PROPOSAL.md) - Performance benchmarks for parallel processing.
5. [**04_VM_SCALING_GUIDE.md**](file:///home/usr_60302531_udst_edu_qa/Kairos_model/audio_singlecall/markdown%20instructions/04_VM_SCALING_GUIDE.md) - RAM engineering and scaling for Azure VMs.
6. [**05_AZURE_DEPLOYMENT.md**](file:///home/usr_60302531_udst_edu_qa/Kairos_model/audio_singlecall/markdown%20instructions/05_AZURE_DEPLOYMENT.md) - Production architecture (Docker + Node.js).
7. [**06_PERFORMANCE_REPORT.md**](file:///home/usr_60302531_udst_edu_qa/Kairos_model/audio_singlecall/markdown%20instructions/06_PERFORMANCE_REPORT.md) - **Active** API-Driven Performance Records.
8. [**06_PERFORMANCE_REPORT_LOCAL_BASELINE.md**](file:///home/usr_60302531_udst_edu_qa/Kairos_model/audio_singlecall/markdown%20instructions/06_PERFORMANCE_REPORT_LOCAL_BASELINE.md) - **Archive**: Full Local Whisper-Small results.
