# Performance Report: High-Quality Audio Pipeline (Azure API)

This report tracks the performance of the **enterprise-grade** audio pipeline using the **Azure OpenAI Whisper API** and **Whisper Medium** (where applicable).

## Current Status: READY FOR BENCHMARKING
> [!IMPORTANT]
> The pipeline has been migrated from local processing to the Azure Whisper API to ensure multilingual script accuracy (Arabic/English) and hallucination-free output.

## High-Quality Benchmark Results (Azure API)

| Video Name | Length | Base ASR | Base AST | API ASR | New AST | Speedup | Quality Note |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Titanic 1997** | 3 h 15m | 3 h 48m | 32.7m | TBD | TBD | - | - |
| **UDST Honors** | 2 h 23m | 4 h 0m* | 45.3m* | TBD | TBD | - | High-Quality Required |
| **DJI Chinatown Walk** | 44.2m | 4 h 0m* | 40.0m* | TBD | TBD | - | - |
| **Learning_ SVM** | 49.6m | 35.0m* | 8.0m* | TBD | TBD | - | - |
| **AI beyond language** | 16.1m | 5.3m | 1.1m | TBD | TBD | - | - |
| **NYC Times Square** | 11.3m | 21.7m | 1.2m | TBD | TBD | - | - |
| **Argentina v France** | 7.7m | 24.9m | 3.2m | TBD | TBD | - | - |
| **How to Make Pasta** | 5.5m | 19.8m | 2.6m | TBD | TBD | - | - |
| **Watch Malala** | 4.6m | 7.3m | 1.0m | TBD | TBD | - | - |
| **Young Sheldon** | 2.8m | 11.5m | 1.5m | TBD | TBD | - | - |

---

## Technical Enhancements in this Version

### 1. Azure OpenAI Whisper API
We have offloaded transcription to the Azure cloud to leverage high-performance compute and enterprise-grade models. This eliminates local memory constraints and provides superior accuracy for diverse languages.

### 2. Global Language Locking
To prevent Whisper from "hallucinating" different languages during background music or applause, we detect the video's primary language once during the pre-scan and **lock** the API to that ISO code.

### 3. Speech Masking for AST Purity
Our **Audio Pre-Scan** now generates a speech-masked buffer. When classifying environmental sounds (AST), we completely zero out regions where people are speaking. This ensures that a dog barking is correctly classified even if someone is talking over it.

### 4. Native Script Preservation
All transcriptions are stored in their native scripts (RTL for Arabic, LTR for English), ensuring the downstream LLM receives contextually accurate data for RAG processing.

---

## Historical Context
For the previous local optimization benchmarks (Whisper-Small), see: 
[**06_PERFORMANCE_REPORT_LOCAL_BASELINE.md**](file:///home/usr_60302531_udst_edu_qa/Kairos_model/audio_singlecall/markdown%20instructions/06_PERFORMANCE_REPORT_LOCAL_BASELINE.md)
