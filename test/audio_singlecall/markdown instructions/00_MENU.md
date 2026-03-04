# 📍 Navigation Menu: Kairos Documentation

This folder contains all the technical guides for the Kairos Video Processing Pipeline. 

### 🟢 Start Here
*   [**01_USER_GUIDE.md**](file:///home/usr_60302531_udst_edu_qa/Kairos_model/audio_singlecall/markdown%20instructions/01_USER_GUIDE.md) - **General Overview.** How to run the pipeline using `./run_pipeline.sh`.

### 📊 Benchmarks & Proposals
*   [**02_HISTORICAL_BENCHMARKS.md**](file:///home/usr_60302531_udst_edu_qa/Kairos_model/audio_singlecall/markdown%20instructions/02_HISTORICAL_BENCHMARKS.md) - **Active** Production API Results.
*   [**02_HISTORICAL_BENCHMARKS_LEGACY_LOCAL.md**](file:///home/usr_60302531_udst_edu_qa/Kairos_model/audio_singlecall/markdown%20instructions/02_HISTORICAL_BENCHMARKS_LEGACY_LOCAL.md) - **Archive**: Legacy Windows vs early Local-Optimized.
*   [**03_PARALLELIZATION_PROPOSAL.md**](file:///home/usr_60302531_udst_edu_qa/Kairos_model/audio_singlecall/markdown%20instructions/03_PARALLELIZATION_PROPOSAL.md) - The case for parallel processing and long-video speedups.

### ⚙️ Scaling & Deployment
*   [**04_VM_SCALING_GUIDE.md**](file:///home/usr_60302531_udst_edu_qa/Kairos_model/audio_singlecall/markdown%20instructions/04_VM_SCALING_GUIDE.md) - RAM engineering and scaling on the Azure 188GB VM.
*   [**05_AZURE_DEPLOYMENT.md**](file:///home/usr_60302531_udst_edu_qa/Kairos_model/audio_singlecall/markdown%20instructions/05_AZURE_DEPLOYMENT.md) - Production architecture (Docker, Node.js, and Redis).
*   [**06_PERFORMANCE_REPORT.md**](file:///home/usr_60302531_udst_edu_qa/Kairos_model/audio_singlecall/markdown%20instructions/06_PERFORMANCE_REPORT.md) - **Active** API-Driven Performance Records.
*   [**06_PERFORMANCE_REPORT_LOCAL_BASELINE.md**](file:///home/usr_60302531_udst_edu_qa/Kairos_model/audio_singlecall/markdown%20instructions/06_PERFORMANCE_REPORT_LOCAL_BASELINE.md) - **Archive**: Full Local Whisper-Small results.
*   [**07_RESUMING_AFTER_SHUTDOWN.md**](file:///home/usr_60302531_udst_edu_qa/Kairos_model/audio_singlecall/markdown%20instructions/07_RESUMING_AFTER_SHUTDOWN.md) - **Fixes & Resuming.**
*   [**09_MANUAL_MIGRATION.md**](file:///home/usr_60302531_udst_edu_qa/Kairos_model/audio_singlecall/markdown%20instructions/09_MANUAL_MIGRATION_TO_MAIN_BRANCH.md) - **Steps to move code to main branch.**
*   [**12_MULTILINGUAL_STRATEGY.md**](file:///home/usr_60302531_udst_edu_qa/Kairos_model/audio_singlecall/markdown%20instructions/12_MULTILINGUAL_TRANSCRIPTION_STRATEGY.md) - **Pros/Cons of Multilingual vs Translated Audio.**
*   [**13_CODEBASE_FEATURES.md**](file:///home/usr_60302531_udst_edu_qa/Kairos_model/audio_singlecall/markdown%20instructions/13_CODEBASE_FEATURES_OVERVIEW.md) - **Overview of Pipeline Python Script Architecture.**
*   [**14_PIPELINE_FLOW.md**](file:///home/usr_60302531_udst_edu_qa/Kairos_model/audio_singlecall/markdown%20instructions/14_PIPELINE_FLOW_EXPLAINED.md) - **Plain-English Step-by-Step Pipeline Walkthrough.**
*   [**15_API_VS_LOCAL.md**](file:///home/usr_60302531_udst_edu_qa/Kairos_model/audio_singlecall/markdown%20instructions/15_API_VS_LOCAL_WHISPER.md) - **Azure API vs Local Whisper Fallback Strategy.**
*   [**16_HALLUCINATION_FILTERING.md**](file:///home/usr_60302531_udst_edu_qa/Kairos_model/audio_singlecall/markdown%20instructions/16_HALLUCINATION_FILTERING.md) - **All 6 Layers of Emoji & Noise Hallucination Filtering.**
*   [**17_NATIVE_LANGUAGE.md**](file:///home/usr_60302531_udst_edu_qa/Kairos_model/audio_singlecall/markdown%20instructions/17_NATIVE_LANGUAGE_TRANSCRIPTION.md) - **Native Language Transcription: Arabic, Chinese, Tagalog, Mixed.**
*   [**18_TIMESTAMP_MATH.md**](file:///home/usr_60302531_udst_edu_qa/Kairos_model/audio_singlecall/markdown%20instructions/18_TIMESTAMP_MAPPING_MATH.md) - **The Mathematical Formulas for Chunking and Scene Mapping.**


---
**Tip:** Use the `./run_pipeline.sh` script in the `audio_singlecall` folder to run the pipeline without memorizing complex flags.
