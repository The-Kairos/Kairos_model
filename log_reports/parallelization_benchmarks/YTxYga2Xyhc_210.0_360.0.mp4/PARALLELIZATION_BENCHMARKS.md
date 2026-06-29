# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 21:51:25 UTC | YTxYga2Xyhc_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 250.190 | 0.774 | 81.434 | 32.904 | 17.331 | 18.698 | 6.505 |

## 2026-06-25 21:51:25 UTC | YTxYga2Xyhc_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/YTxYga2Xyhc_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `250.190` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.774 |
| save_clips | - |
| sample_frames | 1.584 |
| caption_frames | 73.884 |
| sample_fps | 2.642 |
| detect_object_yolo | 13.035 |
| audio_scan | 13.917 |
| asr_timings | 12.170 |
| ast_timings | 55.338 |
| describe_scenes | 32.904 |
| summarize_scenes | 17.331 |
| synthesize_synopsis | 18.698 |
| make_embedding | 6.505 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 75.473 |
| branch_yolo_total | 15.683 |
| branch_audio_total | 81.434 |
