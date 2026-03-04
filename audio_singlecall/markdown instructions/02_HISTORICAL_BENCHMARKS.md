# Historical Benchmarks: High-Quality Production Pipeline

This document compares the legacy Windows-based sequential pipeline with our final high-quality **Azure Whisper API** production configuration.

## Active Benchmark Results (Azure OpenAI API)

| Video Name | Length | Legacy Total (Baseline) | New Total (Azure API) | Speedup | Quality Improvement |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Argentina v France** | 7.7m | 28.5m | TBD | - | Hallucination Free |
| **How to Make Pasta** | 5.5m | 22.6m | TBD | - | - |
| **Watch Malala** | 4.6m | 8.6m | TBD | - | Native Script Accuracy |
| **Young Sheldon** | 2.8m | 13.1m | TBD | - | - |

---

## Technical Context
We transitioned from local model processing (Whisper-Small) to the Azure OpenAI Whisper API to resolve accuracy issues with multilingual content and eliminate phonetic hallucinations in noisy environments.

## Archived Results
To view benchmarks for the initial **Local Whisper (Small)** optimization phase, see:
[**02_HISTORICAL_BENCHMARKS_LEGACY_LOCAL.md**](file:///home/usr_60302531_udst_edu_qa/Kairos_model/audio_singlecall/markdown%20instructions/02_HISTORICAL_BENCHMARKS_LEGACY_LOCAL.md)
