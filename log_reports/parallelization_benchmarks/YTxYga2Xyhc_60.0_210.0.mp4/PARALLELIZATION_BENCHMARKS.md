# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 21:57:46 UTC | YTxYga2Xyhc_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 193.217 | 0.786 | 68.942 | 20.289 | 11.505 | 10.339 | 5.787 |

## 2026-06-25 21:57:46 UTC | YTxYga2Xyhc_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/YTxYga2Xyhc_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `193.217` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.786 |
| save_clips | - |
| sample_frames | 1.220 |
| caption_frames | 59.217 |
| sample_fps | 2.411 |
| detect_object_yolo | 11.271 |
| audio_scan | 15.121 |
| asr_timings | 9.617 |
| ast_timings | 44.196 |
| describe_scenes | 20.289 |
| summarize_scenes | 11.505 |
| synthesize_synopsis | 10.339 |
| make_embedding | 5.787 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 60.443 |
| branch_yolo_total | 13.688 |
| branch_audio_total | 68.942 |
