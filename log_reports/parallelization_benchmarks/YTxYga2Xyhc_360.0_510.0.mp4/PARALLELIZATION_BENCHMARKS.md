# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 21:54:32 UTC | YTxYga2Xyhc_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 186.150 | 0.798 | 67.134 | 21.388 | 11.908 | 9.216 | 5.070 |

## 2026-06-25 21:54:32 UTC | YTxYga2Xyhc_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/YTxYga2Xyhc_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `186.150` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.798 |
| save_clips | - |
| sample_frames | 1.316 |
| caption_frames | 54.227 |
| sample_fps | 2.467 |
| detect_object_yolo | 11.139 |
| audio_scan | 15.351 |
| asr_timings | 10.804 |
| ast_timings | 40.970 |
| describe_scenes | 21.388 |
| summarize_scenes | 11.908 |
| synthesize_synopsis | 9.216 |
| make_embedding | 5.070 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 55.548 |
| branch_yolo_total | 13.612 |
| branch_audio_total | 67.134 |
